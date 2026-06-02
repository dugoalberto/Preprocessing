import os
import re
import argparse
import shutil
import threading
from multiprocessing import Process, Queue, set_start_method
import threading
import queue as thread_queue  # add this import at the top of the file
import numpy as np
from einops import rearrange
from tqdm import tqdm
import torch
import cv2

from utils.image_rescaler import rescale_and_crop
from utils.CLIP import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.feature_extractor import FeatureExtractor
from utils.sam import SAMProcessor

import os
import numpy as np
from huggingface_hub import HfApi

PREPROCESS_DIR = "/tmp/dataset/frames"
SENTINEL = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]
def do_save(save_folder, directory, file_stem, encoder, sam_result, feat_result):
    """Pure I/O, runs in saver thread."""
    sam_path = os.path.join(save_folder, directory, 'SAM', file_stem + '.npy')
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    np.save(sam_path, sam_result)

    feat_dir = os.path.join(save_folder, directory, 'features', encoder)
    os.makedirs(feat_dir, exist_ok=True)
    save_path = os.path.join(feat_dir, file_stem)
    np.save(save_path + '_feats.npy',   feat_result['feats'])
    np.save(save_path + '_seg_map.npy', feat_result['seg_maps'])
    #np.save(save_path + '_feat_map.npy', feat_result['feat_map'])

def load_intrinsics(intrinsic_path: str, orig_w: int, orig_h: int) -> torch.Tensor:
    """
    Legge intrinsic_color.txt (matrice 4x4 pixel-space ScanNet)
    e restituisce una matrice 3x3 normalizzata (fx/w, fy/h, cx/w, cy/h).
    """
    mat = np.loadtxt(intrinsic_path)   # [4, 4]
    fx, fy = mat[0, 0], mat[1, 1]
    cx, cy = mat[0, 2], mat[1, 2]
    return torch.tensor([
        [fx / orig_w,  0.0,          cx / orig_w],
        [0.0,          fy / orig_h,  cy / orig_h],
        [0.0,          0.0,          1.0         ],
    ], dtype=torch.float32)

def save_intrinsics(intrinsic_path: str, intrinsics_norm: torch.Tensor,
                    new_w: int, new_h: int) -> None:
    """
    Riconverte le intrinsics normalizzate in pixel-space e
    riscrive intrinsic_color.txt nel formato 4x4 ScanNet.
    """
    intr = intrinsics_norm
    fx = intr[0, 0].item() * new_w
    fy = intr[1, 1].item() * new_h
    cx = intr[0, 2].item() * new_w
    cy = intr[1, 2].item() * new_h
    mat = np.array([
        [fx,  0.,  cx,  0.],
        [0.,  fy,  cy,  0.],
        [0.,  0.,  1.,  0.],
        [0.,  0.,  0.,  1.],
    ])
    np.savetxt(intrinsic_path, mat, fmt="%.6f")


REPO_ID = "dugoalberto/Scannet_Clip"
api = HfApi(token="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")

def do_save(save_folder, directory, file_stem, encoder, sam_result, feat_result):
    """Pure I/O, runs in saver thread. Salva solo su disco."""

    sam_path = os.path.join(save_folder, directory, 'SAM', file_stem + '.npy')
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    np.save(sam_path, sam_result)

    feat_dir = os.path.join(save_folder, directory, 'features', encoder)
    os.makedirs(feat_dir, exist_ok=True)
    save_path = os.path.join(feat_dir, file_stem)

    np.save(save_path + '_feats.npy', feat_result['feats'])
    np.save(save_path + '_seg_map.npy', feat_result['seg_maps'])

def get_last_completed_stem(save_folder: str, directory: str, encoder: str) -> str | None:
    """
    Returns the file stem of the last successfully processed frame for this scene,
    by looking at which *_feats.npy files exist in features/{encoder}/.
    Returns None if nothing has been processed yet.
    """
    feat_dir = os.path.join(save_folder, directory, "features", encoder)
    if not os.path.exists(feat_dir):
        return None

    npy_files = [f for f in os.listdir(feat_dir) if f.endswith("_seg_map.npy")]
    if not npy_files:
        return None

    # Sort by natural order and take the last one
    npy_files.sort(key=natural_sort_key)
    last_file = npy_files[-1]
    # Strip ".npy" suffix to recover the original stem
    stem = last_file[: -len("_seg_map.npy")]
    return stem


def saver_thread(save_queue):
    while True:
        item = save_queue.get()
        if item is None:
            break
        save_fn, args = item
        try:
            save_fn(*args)
            print(f"[Saver] Saved OK: {args[2]}")  # file_stem
        except Exception as e:
            print(f"[Saver] ERROR: {e}", flush=True)


def gpu_worker(worker_id: int, device_id: int, queue: Queue, sam_ckpt_path: str,
               encoder: str, empty_bg: bool, save_folder: str):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam_processor = SAMProcessor(sam_ckpt_path=sam_ckpt_path, device=device)
    feature_extractors: dict[str, FeatureExtractor] = {}

    # One background saver thread per GPU worker
    save_queue = thread_queue.Queue()
    #save_queue = queue.Queue()  # use threading.Queue, not multiprocessing.Queue
    saver = threading.Thread(target=saver_thread, args=(save_queue,), daemon=True)
    saver.start()

    print(f"[Worker {worker_id}] Ready on {device}")

    while True:
        item = queue.get()
        if item is SENTINEL:
            save_queue.put(None)  # shutdown saver
            saver.join()
            print(f"[Worker {worker_id}] Shutting down.")
            break

        directory, file_stem, image, new_w, new_h = item

        if directory not in feature_extractors:
            feature_extractors[directory] = FeatureExtractor(
                save_folder + "/" + directory, model
            )

        with torch.no_grad():
            # Returns results instead of saving internally
            sam_result = sam_processor.process_images_nosave(
                image, file_stem, f"{save_folder}/{directory}", empty_bg=empty_bg
            )
            feat_result = feature_extractors[directory].create_features_nosave(
                sam_result, method=encoder, level='l'
            )

        # Offload all saving to background thread
        save_queue.put((do_save, (save_folder, directory, file_stem, encoder,
                                  sam_result, feat_result)))



def io_producer(data_list: list, dataset_dir: str, save_folder: str,
                args, queues: list[Queue]):
    worker_count = len(queues)
    worker_idx = 0
    shape = (args.resolution, args.resolution)   # (H_out, W_out)
    h_out, w_out = shape

    for directory in tqdm(data_list, desc="Scenes", ascii=True):
        img_folder      = os.path.join(dataset_dir, directory, "iphone", "rgb")
        intrinsic_path  = os.path.join(dataset_dir, directory, "intrinsic", "intrinsic_color.txt")

        if not os.path.exists(img_folder):
            continue

        directory_data_list = sorted(os.listdir(img_folder), key=natural_sort_key)

        # --- Resume logic ---
        last_done = get_last_completed_stem(save_folder, directory, args.encoder)
        if last_done is not None:
            # Find the index of the last completed frame in the sorted file list
            stems = [f.split('.')[0] for f in directory_data_list]
            if last_done in stems:
                resume_idx = stems.index(last_done) + 1
                print(f"[Resume] Scene '{directory}': skipping {resume_idx} frame(s), "
                      f"resuming after '{last_done}'")
                directory_data_list = directory_data_list[resume_idx:]
            else:
                print(f"[Resume] Scene '{directory}': last stem '{last_done}' not found, "
                      f"starting from scratch.")
        else:
            print(f"[Resume] Scene '{directory}': starting from scratch.")

        if not directory_data_list:
            print(f"[Resume] Scene '{directory}': already fully processed, skipping.")
            continue

        # ── Leggi un frame per ricavare orig_w, orig_h ──────────────────
        first_frame = cv2.imread(os.path.join(img_folder, directory_data_list[0]))
        orig_h, orig_w = first_frame.shape[:2]

        # ── Carica e aggiorna le intrinsics (una volta per scena) ────────
        if os.path.exists(intrinsic_path):
            intrinsics_norm = load_intrinsics(intrinsic_path, orig_w, orig_h)
            # rescale_and_crop aggiorna solo l'immagine qui, le intrinsics le gestiamo
            # calcolando row/col manualmente per poterle salvare
            scale       = max(h_out / orig_h, w_out / orig_w)
            h_sc        = round(orig_h * scale)
            w_sc        = round(orig_w * scale)
            row         = (h_sc - h_out) // 2
            col         = (w_sc - w_out) // 2

            # step 1: rescale
            intr = intrinsics_norm.clone()
            intr[0, 0] *= w_sc / orig_w;  intr[0, 2] *= w_sc / orig_w
            intr[1, 1] *= h_sc / orig_h;  intr[1, 2] *= h_sc / orig_h
            # step 2: crop
            intr[0, 2] = (intr[0, 2] * w_sc - col) / w_out
            intr[1, 2] = (intr[1, 2] * h_sc - row) / h_out
            intr[0, 0] *= w_sc / w_out
            intr[1, 1] *= h_sc / h_out

            # Riscrivi il file con le intrinsics aggiornate
            save_intrinsics(intrinsic_path, intr, w_out, h_out)
            print(f"[Intrinsics] Updated and saved for scene '{directory}'")
        else:
            print(f"[Intrinsics] intrinsic_color.txt not found for '{directory}', skipping.")
            intrinsics_norm = None

        # ── resolution.txt ───────────────────────────────────────────────
        res_file = os.path.join(save_folder, "resolution.txt")
        if not os.path.exists(res_file):
            with open(res_file, "w") as f:
                f.write(f"{w_out} {h_out}\n")

        print(len(directory_data_list))
        for file_name in tqdm(directory_data_list, desc=f"Reading {directory}", ascii=True):
            bgr = cv2.imread(os.path.join(img_folder, file_name))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(rgb / 255.0).float()
            img_t = rearrange(img_t, "h w c -> c h w")

            # usa intrinsics dummy se il file non c'era
            intr_in = intrinsics_norm if intrinsics_norm is not None else torch.eye(3)
            img_crop, _ = rescale_and_crop(img_t, intr_in, shape)

            img_out = (rearrange(img_crop, "c h w -> h w c").numpy() * 255).astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

            file_stem = file_name.split('.')[0]
            queues[worker_idx % worker_count].put(
                (directory, file_stem, img_out, w_out, h_out)
            )

        worker_idx += 1

        for subdir in ["tiles", "SAM_vis", "SAM"]:
            path = os.path.join(PREPROCESS_DIR, directory, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)
        scene_local_path = os.path.join(save_folder, directory)
        scene_repo_path = directory  # path dentro il repo

        try:
            api.upload_folder(
                folder_path=scene_local_path,
                path_in_repo=scene_repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"[HF] Upload completato: {directory}")
            shutil.rmtree(scene_local_path)  # libera disco dopo upload
        except Exception as e:
            print(f"[HF] Errore upload {directory}: {e}")

    for q in queues:
        q.put(SENTINEL)


if __name__ == '__main__':
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str,
                        default="/mnt/home/albertodugo/Projects/Preproccessing/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder', type=str, default="clip")
    parser.add_argument('--empty_bg', action='store_true', default=False)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--workers_per_gpu', type=int, default=1)
    args = parser.parse_args()

    save_folder = "/tmp/dataset/frames/"
    dataset_dir = "/tmp/dataset/frames/"
    data_list = sorted(os.listdir(dataset_dir))

    print(f"Launching with {args.num_gpus} GPU(s), {args.workers_per_gpu} worker(s) per GPU")

    queue_maxsize = 8
    queues = [Queue(maxsize=queue_maxsize)
              for _ in range(args.num_gpus * args.workers_per_gpu)]

    workers = []
    for gpu_id in range(args.num_gpus):
        for w in range(args.workers_per_gpu):
            worker_id = gpu_id * args.workers_per_gpu + w
            p = Process(
                target=gpu_worker,
                args=(worker_id, gpu_id, queues[worker_id],
                      args.sam_ckpt_path, args.encoder, args.empty_bg, save_folder)
            )
            p.start()
            workers.append(p)

    io_producer(data_list, dataset_dir, save_folder, args, queues)

    for p in workers:
        p.join()

    print("All done.")
