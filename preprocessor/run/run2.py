
import os
import re
import shutil
from io import BytesIO
from pathlib import Path

import cv2
import argparse

from PIL import Image
from tqdm import tqdm
import torch
from feature_extractor import FeatureExtractor

import cv2

from CLIP import OpenCLIPNetwork, OpenCLIPNetworkConfig

cv2.setNumThreads(1) # CRITICAL: Tells OpenCV to stop spamming threads

# ... rest of your imports ...

if __name__ == '__main__':
    #pip install open-clip-torch
    #cd segment-anything-langsplat
    #pip install -e .
    #cd ../prerocessor/
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="/mnt/home/albertodugo/Projects/Preproccessing/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder', type=str, default="clip")
    parser.add_argument('--empty_bg', action='store_true', default=False)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_dir = "/Preproccessing/Datasets/re10k/try"  # Change this to the path where your .torch files are
    img_folder = os.path.join(dataset_dir, 'images_test_0_2')
    save_folder = os.path.join(dataset_path, '../preprocess_test')
    data_list=sorted(os.listdir(img_folder))
    print(f"Image segmentation, method: SAM(langsplat), empty background: {args.empty_bg}")
    from sam import SAMProcessor

    sam_processor = SAMProcessor(sam_ckpt_path=args.sam_ckpt_path, device='cuda')
    for directory in tqdm(data_list, desc="Processing files"):
        print(f"Processing {directory}")
        img_folder_1 = os.path.join(img_folder, directory)
        WARNED = False
        if not os.path.exists(img_folder_1):
            continue
        directory_data_list = os.listdir(img_folder_1)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]
        directory_data_list.sort(key=natural_sort_key)
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        sam_processor = SAMProcessor(sam_ckpt_path=args.sam_ckpt_path, device='cuda')
        for directory in tqdm(data_list, desc="Processing files"):
            print(f"Processing {directory}")
            img_folder_1 = os.path.join(img_folder, directory)
            WARNED = False
            if not os.path.exists(img_folder_1):
                continue
            directory_data_list = os.listdir(img_folder_1)


            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', s)]


            directory_data_list.sort(key=natural_sort_key)
            for file_name in tqdm(directory_data_list, desc="Processing files"):
                with torch.no_grad():
                    image_path = os.path.join(img_folder_1, file_name)
                    image = cv2.imread(image_path)
                    orig_h, orig_w = image.shape[:2]
                    print(file_name)
                    if args.resolution == -1:
                        if orig_h > 1080 and not WARNED:
                            print("[ INFO ] Large image detected (>1080P), rescaling to 1080P.")
                            WARNED = True
                        scale = orig_h / 1080 if orig_h > 1080 else 1
                    else:
                        scale = orig_w / args.resolution

                    new_h, new_w = int(orig_h / scale), int(orig_w / scale)
                    image = cv2.resize(image, (new_w, new_h))
                    # image = torch.from_numpy(image)

                    with open(f"{save_folder}/resolution.txt", "w") as f:
                        f.write(f"{new_w} {new_h}\n")

                    sam_processor.process_images(image, file_name.split('.')[0], save_folder + "/" + directory,
                                                 empty_bg=args.empty_bg)
                feature_extractor = FeatureExtractor(save_folder + "/" + directory, model)
                with torch.no_grad():
                    feature_extractor.create_features(file_name.split('.')[0], method=args.encoder, level='l')
                if os.path.exists(
                        f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/tiles/"):
                    path = f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/tiles/"
                    shutil.rmtree(path)
                if os.path.exists(
                        f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/SAM_vis/"):
                    path = f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/SAM_vis/"
                    shutil.rmtree(path)
                if os.path.exists(
                        f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/SAM/"):
                    path = f"/mnt/home/albertodugo/Projects/Preproccessing/preprocessor/preprocess_test/{directory}/SAM/"
                    shutil.rmtree(path)
