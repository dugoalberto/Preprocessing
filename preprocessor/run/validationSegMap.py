"""
===============================================================================
Guarda se le segmentazioni sono tutte presenti, oppure c'è qualche mismatch
===============================================================================

Questo script esegue un controllo di integrità (sanity check) sul dataset re10k
per verificare che il preprocessing delle immagini sia andato a buon fine.

Cosa fa nel dettaglio:
1. LETTURA: Cerca e carica in sequenza tutti i file `.torch` nella cartella di
   train, che contengono i metadati e le liste delle immagini per ogni scena.
2. MATCHING: Per ogni scena trovata nei file `.torch`, controlla se esiste
   una cartella corrispondente nella directory di test/try.
3. VALIDAZIONE: Se la directory della scena esiste, verifica che per OGNI
   immagine siano stati correttamente salvati sul disco:
   - Le feature di CLIP (file `[idx]_feats.npy`)
   - Le mappe di segmentazione (file `[idx]_seg_map.npy`)
4. REPORT: Stampa a schermo un log in tempo reale per segnalare esattamente
   quali file mancano e genera un Summary finale con le statistiche totali
   di scene e immagini processate, trovate o mancanti.
===============================================================================
"""

import os
import torch
from pathlib import Path

dataset_dir = "/Preproccessing/Datasets/re10k/try"
img_folder = os.path.join(dataset_dir, "images_test_0_2")

torch_files = list(Path("/Datasets/re10k/train").rglob("*.torch"))

try_base = Path("/Datasets/re10k/try")

# Counters for summary
total_scenes = 0
scenes_found = 0
scenes_missing = 0
total_images = 0
images_ok = 0
images_missing_feats = 0
images_missing_seg = 0
images_missing_both = 0

for tf in torch_files:
    print(f"\nProcessing {tf.name}...")

    data = torch.load(tf, map_location="cpu", weights_only=False)

    if not isinstance(data, list):
        print(f"  Skipping {tf.name}: Expected a list of scenes, got {type(data)}.")
        continue

    for scene_idx, scene_dict in enumerate(data):
        if not isinstance(scene_dict, dict) or "images" not in scene_dict:
            continue

        scene_key = scene_dict.get("key", f"unknown_scene_{scene_idx}")
        total_scenes += 1

        # Skip scenes not present in try/ — not a problem
        scene_dir = try_base / str(scene_key)
        if not scene_dir.exists():
            scenes_missing += 1
            continue

        scenes_found += 1
        images = scene_dict["images"]  # shape: (N, C, H, W) or list of N images

        n_images = len(images)
        total_images += n_images

        clip_dir = scene_dir / "features" / "clip"

        for img_idx in range(n_images):
            feats_path = clip_dir / f"{img_idx}_feats.npy"
            seg_path   = clip_dir / f"{img_idx}_seg_map.npy"

            has_feats = feats_path.exists()
            has_seg   = seg_path.exists()

            if has_feats and has_seg:
                images_ok += 1
            elif not has_feats and not has_seg:
                images_missing_both += 1
                print(f"  [MISSING BOTH]  scene={scene_key}  img={img_idx}")
            elif not has_feats:
                images_missing_feats += 1
                print(f"  [MISSING FEATS] scene={scene_key}  img={img_idx}  ({feats_path.name})")
            else:
                images_missing_seg += 1
                print(f"  [MISSING SEG]   scene={scene_key}  img={img_idx}  ({seg_path.name})")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Torch files processed : {len(torch_files)}")
print(f"  Total scenes          : {total_scenes}")
print(f"    ✓ Found in try/     : {scenes_found}")
print(f"    ✗ Missing           : {scenes_missing}")
print(f"  Total images checked  : {total_images}")
print(f"    ✓ Both files OK     : {images_ok}")
print(f"    ✗ Missing feats     : {images_missing_feats}")
print(f"    ✗ Missing seg_map   : {images_missing_seg}")
print(f"    ✗ Missing both      : {images_missing_both}")
print("="*60)