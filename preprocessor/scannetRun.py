import os
import re
import argparse
import shutil
import threading
from multiprocessing import Process, Queue, set_start_method
import threading
import queue as thread_queue  # add this import at the top of the file
import numpy as np
from tqdm import tqdm
import torch
import cv2

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



REPO_ID = "dugoalberto/Scannet_Clip"
api = HfApi(token="..")

def do_save(save_folder, directory, file_stem, encoder, sam_result, feat_result):
    """Pure I/O, runs in saver thread. Salva su disco e fa l'upload su Hugging Face."""

    sam_path = os.path.join(save_folder, directory, 'SAM', file_stem + '.npy')
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    np.save(sam_path, sam_result)

    feat_dir = os.path.join(save_folder, directory, 'features', encoder)
    os.makedirs(feat_dir, exist_ok=True)
    save_path = os.path.join(feat_dir, file_stem)

    feats_file = save_path + '_feats.npy'
    seg_map_file = save_path + '_seg_map.npy'

    np.save(feats_file, feat_result['feats'])
    np.save(seg_map_file, feat_result['seg_maps'])

    repo_sam_path = f"{directory}/SAM/{file_stem}.npy"
    repo_feats_path = f"{directory}/features/{encoder}/{file_stem}_feats.npy"
    repo_seg_maps_path = f"{directory}/features/{encoder}/{file_stem}_seg_map.npy"

    try:
        # Carica il file SAM
        api.upload_file(
            path_or_fileobj=sam_path,
            path_in_repo=repo_sam_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            silent=True
        )
        # Carica il file Feats
        api.upload_file(
            path_or_fileobj=feats_file,
            path_in_repo=repo_feats_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            silent=True
        )
        api.upload_file(
            path_or_fileobj=seg_map_file,
            path_in_repo=repo_seg_maps_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            silent=True
        )
        os.remove(sam_path)
        os.remove(feats_file)
        os.remove(seg_map_file)

        # Crea dei file vuoti (0 byte) con lo stesso nome
        # Questo permette a `get_last_completed_stem` di funzionare per il resume!
        open(sam_path, 'w').close()
        open(feats_file, 'w').close()
        open(seg_map_file, 'w').close()
    except Exception as e:
        print(f"\n[Errore Upload HF] Fallito l'upload per {file_stem}: {e}")

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
    stem = last_file[: -len(".npy")]
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

    for directory in tqdm(data_list, desc="Scenes", ascii=True):
        img_folder = os.path.join(dataset_dir, directory, "iphone", "rgb")
        if not os.path.exists(img_folder):
            continue

        directory_data_list = sorted(os.listdir(img_folder), key=natural_sort_key)

        # --- Resume logic ---
        last_done = get_last_completed_stem(save_folder, directory, args.encoder)
        if last_done is not None:
            # Find the index of the last completed frame in the sorted file list
            stems = [f.split('.')[0] for f in directory_data_list]
            if last_done in stems:
                resume_idx = stems.index(last_done) + 1  # start from the next one
                print(f"[Resume] Scene '{directory}': skipping {resume_idx} frame(s), "
                      f"resuming after '{last_done}'")
                directory_data_list = directory_data_list[resume_idx:]
            else:
                print(f"[Resume] Scene '{directory}': last stem '{last_done}' not found "
                      f"in image list, starting from scratch.")
        else:
            print(f"[Resume] Scene '{directory}': no previous features found, "
                  f"starting from scratch.")

        if not directory_data_list:
            print(f"[Resume] Scene '{directory}': already fully processed, skipping.")
            continue

        scale = None
        new_w = new_h = None
        WARNED = False
        print(len(directory_data_list))
        for file_name in tqdm(directory_data_list, desc=f"Reading {directory}", ascii=True):
            image = cv2.imread(os.path.join(img_folder, file_name))
            orig_h, orig_w = image.shape[:2]
            if scale is None:
                if args.resolution == -1:
                    if orig_h > 1080 and not WARNED:
                        print("[ INFO ] Large image detected (>1080P), rescaling to 1080P.")
                        WARNED = True
                    scale = orig_h / 1080 if orig_h > 1080 else 1
                else:
                    scale = orig_w / args.resolution
                new_h, new_w = int(orig_h / scale), int(orig_w / scale)
                # Only write resolution.txt if not already there
                res_file = f"{save_folder}/resolution.txt"
                if not os.path.exists(res_file):
                    with open(res_file, "w") as f:
                        f.write(f"{new_w} {new_h}\n")

            image = cv2.resize(image, (new_w, new_h))
            file_stem = file_name.split('.')[0]
            queues[worker_idx % worker_count].put((directory, file_stem, image, new_w, new_h))
        worker_idx += 1

        for subdir in ["tiles", "SAM_vis", "SAM"]:
            path = os.path.join(PREPROCESS_DIR, directory, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)

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

    save_folder = "/tmp/dataset/frames/frames"
    dataset_dir = "/tmp/dataset/frames/frames"
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
