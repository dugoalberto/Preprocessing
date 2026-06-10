import os
import numpy as np

base_dir = '/tmp/dataset/frames'

print(f"{'SCENA':<15} | {'STATO':<10} | {'EXTRINSICS':<12} | {'FRAMES':<10}")
print("-" * 55)

# Cerca tutte le cartelle delle scene
for scene_id in sorted(os.listdir(base_dir)):
    scene_path = os.path.join(base_dir, scene_id)

    # Salta se non è una cartella
    if not os.path.isdir(scene_path):
        continue

    extrinsics_file = os.path.join(scene_path, 'extrinsics.npy')
    rgb_dir = os.path.join(scene_path, 'iphone', 'rgb')

    # Controlla che i file/cartelle esistano
    if not os.path.exists(extrinsics_file) or not os.path.exists(rgb_dir):
        print(f"{scene_id:<15} | {'ERRORE':<10} | File o cartelle mancanti")
        continue

    try:
        # 1. Ottieni il numero di elementi nel file npy (assumendo che la prima dimensione sia il numero di frame)
        extrinsics_data = np.load(extrinsics_file)
        num_extrinsics = extrinsics_data.shape[0]

        # 2. Conta i file immagine nella cartella rgb (filtriamo per estensione per sicurezza)
        valid_exts = ('.jpg', '.jpeg', '.png')
        num_frames = sum(1 for f in os.listdir(rgb_dir) if f.lower().endswith(valid_exts))

        # 3. Confronta
        if num_extrinsics == num_frames:
            status = "OK"
        else:
            status = "MISMATCH"

        print(f"{scene_id:<15} | {status:<10} | {num_extrinsics:<12} | {num_frames:<10}")

    except Exception as e:
        print(f"{scene_id:<15} | {'ERRORE':<10} | Impossibile leggere: {e}")