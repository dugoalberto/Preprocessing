"""
===============================================================================
Script di Estrazione Immagini da File .torch (Dataset re10k)
===============================================================================

Questo script si occupa di spacchettare (unpack) i dati visivi salvati
all'interno dei file `.torch` del dataset e di esportarli come normali
file `.jpg` sul disco, mantenendo l'organizzazione per scene.

Cosa fa nel dettaglio:
1. LETTURA: Cerca tutti i file `.torch` nella cartella di train e ne carica
   il contenuto (che si aspetta essere una lista di dizionari, uno per scena).
2. CREAZIONE CARTELLE: Per ogni scena individuata, estrae la sua `key`
   identificativa e crea una sottocartella dedicata nella directory di output.
3. DECODIFICA (Nel blocco commentato): Scorre i dati delle immagini della
   scena, gestendo intelligentemente due formati:
   - Byte-encoded (Tensor 1D): decodificati al volo tramite `PIL` e `BytesIO`.
   - Raw Tensor: normalizzati (0-1) e riordinati nei canali corretti (C, H, W).
4. ESPORTAZIONE (Nel blocco commentato): Salva i frame elaborati in formato
   `.jpg` (es. `0.jpg`, `1.jpg`...) dentro la cartella della rispettiva scena.
===============================================================================
"""

import os
import io
import torch
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

def extractImagefromTorch(
        dataset_dir="/mnt/home/albertodugo/Projects/Preproccessing/Datasets/re10k/"):
    images_name_folder = "images_test_0_2"
    img_folder = os.path.join(dataset_dir,"try" ,images_name_folder)
    os.makedirs(img_folder, exist_ok=True)
    torch_files = list(Path(dataset_dir+"train").rglob("*.torch"))

    for tf in torch_files:
        print(f"Processing {tf.name}...")

        # Load the file: This returns a list of scene dictionaries
        data = torch.load(tf, map_location="cpu", weights_only=False)

        if not isinstance(data, list):
            print(f"  Skipping {tf.name}: Expected a list of scenes, got {type(data)}.")
            continue

        # 1. Outer Loop: Iterate through each scene in the file
        for scene_idx, scene_dict in enumerate(data):
            if not isinstance(scene_dict, dict) or "images" not in scene_dict:
                continue
            scene_key = scene_dict.get("key", f"unknown_scene_{scene_idx}")
            print(scene_key)
            # Get the folder name from the 'key'
            scene_key = scene_dict.get("key", f"unknown_scene_{scene_idx}")
            scene_dir = os.path.join(img_folder, str(scene_key))
            os.makedirs(scene_dir, exist_ok=True)

            images = scene_dict["images"]
            saved_count = 0
            # Handle whether 'images' is a list of items or a single batched tensor
            if isinstance(images, torch.Tensor):
                image_iterable = images
            elif isinstance(images, list):
                image_iterable = images
            else:
                print(f"  Skipping images in {scene_key}: Unknown format {type(images)}")
                continue

            # 2. Inner Loop: Iterate through the images for this specific scene
            for i, img_data in enumerate(image_iterable):
                try:
                    # Handle Byte-encoded (1D Tensor)
                    if isinstance(img_data, torch.Tensor) and img_data.ndim == 1:
                        img = Image.open(io.BytesIO(img_data.numpy().tobytes()))
                        img_tensor = to_tensor(img)

                    # Handle Raw Tensors
                    elif isinstance(img_data, torch.Tensor):
                        img_tensor = img_data.float()
                        if img_tensor.max() > 1.0:
                            img_tensor /= 255.0
                        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                            img_tensor = img_tensor.permute(2, 0, 1)
                    else:
                        print(f"    Skipping frame {i}: Unrecognized image data type.")
                        continue

                    # Save as 0.jpg, 1.jpg, etc., inside the scene's folder
                    save_path = os.path.join(scene_dir, f"{saved_count}.jpg")
                    save_image(img_tensor, save_path)
                    saved_count += 1

                except Exception as e:
                    print(f"  Error on scene {scene_key}, frame {i}: {e}")

            print(f"  Done. Saved {saved_count} images to folder: {scene_key}")

#this is used for look at the scene into the torch files
def readScenesfromTorch(
        dataset_dir="/mnt/home/albertodugo/Projects/Preproccessing/Datasets/re10k/"):
    images_name_folder = "images_test_0_2"
    img_folder = os.path.join(dataset_dir,"try" ,images_name_folder)
    os.makedirs(img_folder, exist_ok=True)
    torch_files = list(Path(dataset_dir+"train").rglob("*.torch"))

    for tf in torch_files:
        print(f"Processing {tf.name}...")

        # Load the file: This returns a list of scene dictionaries
        data = torch.load(tf, map_location="cpu", weights_only=False)

        if not isinstance(data, list):
            print(f"  Skipping {tf.name}: Expected a list of scenes, got {type(data)}.")
            continue

        # 1. Outer Loop: Iterate through each scene in the file
        for scene_idx, scene_dict in enumerate(data):
            if not isinstance(scene_dict, dict) or "images" not in scene_dict:
                continue
            scene_key = scene_dict.get("key", f"unknown_scene_{scene_idx}")
            print(scene_key)


if __name__ == '__main__':
    extractImagefromTorch()
    readScenesfromTorch()