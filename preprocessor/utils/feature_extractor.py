import os
import random
import numpy as np
import torch
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image

class FeatureExtractor:
    def __init__(self, save_folder, model, seed=42):
        self.model = model
        self.save_folder = save_folder
        self.seed = seed

        self._seed_everything(self.seed)
        torch.set_default_dtype(torch.float32)


    def _seed_everything(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    def create_features(self, name, method='clip', level='l'):
        try:
            sam_path = os.path.join(self.save_folder, 'SAM', name + '.npy')
            data = np.load(sam_path, allow_pickle=True).item()
            seg_images = data["seg_images"]
            seg_maps = data["seg_maps"]

            # DEBUG 1: controlla dtype salvato nel .npy
            print(f"[DEBUG] seg_images['{level}'] dtype from npy: {seg_images[level].dtype}")

            seg_map = torch.from_numpy(seg_maps[level]).to("cuda")
            tiles = torch.from_numpy(seg_images[level]).float().to("cuda")

            # DEBUG 2: controlla dtype tiles dopo cast
            print(f"[DEBUG] tiles dtype after .float(): {tiles.dtype}")

            # DEBUG 3: controlla dtype dei pesi del modello
            for pname, param in self.model.named_parameters():
                print(f"[DEBUG] first model param '{pname}' dtype: {param.dtype}")
                break

            with torch.no_grad():
                feat = self.model.encode_image(tiles)

                # DEBUG 4: controlla dtype di feat
                print(f"[DEBUG] feat dtype after encode_image: {feat.dtype}")

                img_embed = feat.detach().cpu()
                feature_map, valid_mask = self.model.get_feature_map(seg_map.to("cuda"), feat.to("cuda"))

            os.makedirs(os.path.join(self.save_folder, 'features', method), exist_ok=True)
            save_path = os.path.join(self.save_folder, 'features', method, name)

            return {
                'feats': img_embed.cpu().numpy(),
                'seg_maps': seg_maps,
                'feat_map': {'feat_map': feature_map.cpu().numpy(),
                             'valid_mask': valid_mask.cpu().numpy()}
            }

        except Exception as e:
            print(f"[ WARNING ] Error embedding image {name}: {e}")
            return

    def create_features_nosave(self, sam_data: dict, method='clip', level='l'):
        try:
            seg_images = sam_data["seg_images"]
            seg_maps = sam_data["seg_maps"]
            tiles = seg_images[level]
            if not isinstance(tiles, torch.Tensor):
                tiles = torch.from_numpy(tiles)
            tiles = tiles.float().to("cuda")

            seg_map = seg_maps[level]
            if not isinstance(seg_map, torch.Tensor):
                seg_map = torch.from_numpy(seg_map)
            seg_map = seg_map.to("cuda")

            with torch.no_grad():
                feat = self.model.encode_image(tiles)

                # DEBUG 4: controlla dtype di feat

                img_embed = feat.detach().cpu()
                feature_map, valid_mask = self.model.get_feature_map(seg_map.to("cuda"), feat.to("cuda"))


            return {
                'feats': img_embed.cpu().numpy(),
                'seg_maps': seg_maps,
                #'feat_map': {'feat_map': feature_map.cpu().numpy(),
                             #'valid_mask': valid_mask.cpu().numpy()}
            }

        except Exception as e:

            import traceback

            print(f"[ ERROR ] create_features_nosave failed: {e}")

            traceback.print_exc()  # this shows the actual line and error

            return None

            

    
