
import os
import re
import argparse
import shutil
from tqdm import tqdm
import torch
from Preproccessing.preprocessor.utils.feature_extractor import FeatureExtractor
from Preproccessing.preprocessor.utils.CLIP import OpenCLIPNetwork, OpenCLIPNetworkConfig

#HOW TO RUN
# pip install open-clip-torch
# git clone https://github.com/minghanqin/segment-anything-langsplat.git
# cd segment-anything-langsplat
# pip install -e .
# cd ../prerocessor/

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_ckpt_path', type=str,
                        default="/mnt/home/albertodugo/Projects/Preproccessing/ckpt/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    dataset_dir = "/Preproccessing/preprocessor"  # Change this to the path where your .torch files are
    img_folder = os.path.join(dataset_dir, "../preprocess")

    data_list=sorted(os.listdir(img_folder))
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    for directory in tqdm(data_list, desc="Processing files"):
        print(f"Processing {directory}")
        img_folder_1 = os.path.join(img_folder, directory, "SAM")
        WARNED = False
        if not os.path.exists(img_folder_1):
            continue
        directory_data_list = os.listdir(img_folder_1)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]
        directory_data_list.sort(key=natural_sort_key)
        for file_name in tqdm(directory_data_list, desc="Processing files"):
            feature_extractor = FeatureExtractor(img_folder+"/"+directory, model)
            with torch.no_grad():
                feature_extractor.create_features(file_name.split('.')[0], method="clip", level='l')
        shutil.rmtree(img_folder_1)
