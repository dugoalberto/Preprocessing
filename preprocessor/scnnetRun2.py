import os
import re
import argparse
import shutil
import threading
from multiprocessing import Process, Queue, set_start_method
import queue as thread_queue
import numpy as np
from einops import rearrange
from tqdm import tqdm
import torch
import cv2

from utils.image_rescaler import rescale_and_crop
from utils.CLIP import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.feature_extractor import FeatureExtractor
from utils.sam import SAMProcessor

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PREPROCESS_DIR = "/tmp/dataset/frames"
SENTINEL = object()  # unique sentinel, not None

REPO_ID  = "dugoalberto/Scannet_Clip"
HF_TOKEN = "-"

BATCH_SCENES  = 1   # how many scenes to download before processing
LOAD_WORKERS  = 16   # threads for parallel image loading inside producer
FRAME_BATCH   = 256  # frames buffered per scene in the producer loop

# ─────────────────────────────────────────────────────────────────────────────# HuggingFace helpers
# ─────────────────────────────────────────────────────────────────────────────
def scene_already_processed(api: HfApi, repo_id: str, scene_name: str) -> bool:
    for subdir in ["features"]:
        try:
            files = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=f"{scene_name}/{subdir}",
            )
            if not any(True for _ in files):
                return False
        except EntryNotFoundError:
            return False
    return True


def download_scene(api: HfApi, repo_id: str, scene_name: str, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[f"{scene_name}/iphone/rgb/*"],
        token=HF_TOKEN,
    )
    return os.path.join(local_dir, scene_name)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def get_last_completed_stem(save_folder: str, directory: str, encoder: str):
    feat_dir = os.path.join(save_folder, directory, "features", encoder)
    if not os.path.exists(feat_dir):
        return None
    npy_files = [f for f in os.listdir(feat_dir) if f.endswith("_seg_map.npy")]
    if not npy_files:
        return None
    npy_files.sort(key=natural_sort_key)
    return npy_files[-1][: -len("_seg_map.npy")]


# ─────────────────────────────────────────────────────────────────────────────
# Saver thread  (inside each GPU worker process)
# ─────────────────────────────────────────────────────────────────────────────
def do_save(save_folder, directory, file_stem, encoder, sam_result, feat_result):
    sam_path = os.path.join(save_folder, directory, 'SAM', file_stem + '.npy')
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    np.save(sam_path, sam_result)

    feat_dir = os.path.join(save_folder, directory, 'features', encoder)
    os.makedirs(feat_dir, exist_ok=True)
    save_path = os.path.join(feat_dir, file_stem)
    np.save(save_path + '_feats.npy',   feat_result['feats'])
    np.save(save_path + '_seg_map.npy', feat_result['seg_maps'])


def saver_thread(save_queue: thread_queue.Queue):
    while True:
        item = save_queue.get()
        if item is None:
            break
        save_fn, args = item
        try:
            save_fn(*args)
        except Exception as e:
            print(f"[Saver] ERROR: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace upload thread  (main process)
# ─────────────────────────────────────────────────────────────────────────────
def hf_upload_thread(upload_queue: thread_queue.Queue):
    api = HfApi(token=HF_TOKEN)
    while True:
        item = upload_queue.get()
        if item is None:
            break
        scene_local_path, scene_repo_path = item
        try:
            api.upload_folder(
                folder_path=scene_local_path,
                path_in_repo=scene_repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["features/**", "SAM/**"],
            )
            print(f"[HF] Upload done: {scene_repo_path}", flush=True)
            shutil.rmtree(scene_local_path, ignore_errors=True)
            print(f"[HF] Cleaned up local: {scene_local_path}", flush=True)
        except Exception as e:
            print(f"[HF] Upload ERROR for {scene_repo_path}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPU worker process
# ─────────────────────────────────────────────────────────────────────────────
def gpu_worker(worker_id: int, device_id: int, in_queue: Queue,
               sam_ckpt_path: str, encoder: str, empty_bg: bool, save_folder: str):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)

    model         = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam_processor = SAMProcessor(sam_ckpt_path=sam_ckpt_path, device=device)
    feature_extractors: dict[str, FeatureExtractor] = {}

    save_queue: thread_queue.Queue = thread_queue.Queue(maxsize=8)
    saver = threading.Thread(target=saver_thread, args=(save_queue,), daemon=True)
    saver.start()

    print(f"[Worker {worker_id}] Ready on {device}", flush=True)

    while True:
        item = in_queue.get()
        if item is SENTINEL:
            in_queue.put(SENTINEL)   # re-enqueue so sibling workers also stop
            save_queue.put(None)
            saver.join()
            print(f"[Worker {worker_id}] Shutdown.", flush=True)
            break

        directory, file_stem, image, new_w, new_h = item

        if directory not in feature_extractors:
            feature_extractors[directory] = FeatureExtractor(
                save_folder + "/" + directory, model
            )

        with torch.no_grad():
            sam_result  = sam_processor.process_images_nosave(
                image, file_stem, f"{save_folder}/{directory}", empty_bg=empty_bg
            )
            feat_result = feature_extractors[directory].create_features_nosave(
                sam_result, method=encoder, level='l'
            )

        save_queue.put((do_save, (save_folder, directory, file_stem, encoder,
                                  sam_result, feat_result)))


# ─────────────────────────────────────────────────────────────────────────────
# Producer  — only called with already-downloaded scenes
# ─────────────────────────────────────────────────────────────────────────────
def load_frame_task(args_tuple):
    """Top-level so it is picklable by ProcessPoolExecutor."""
    file_name, img_folder, shape = args_tuple
    bgr = cv2.imread(os.path.join(img_folder, file_name))
    if bgr is None:
        return None, None
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_t   = rearrange(torch.from_numpy(rgb / 255.0).float(), "h w c -> c h w")
    img_crop = rescale_and_crop(img_t, shape)
    img_out  = (rearrange(img_crop, "c h w -> h w c").numpy() * 255).astype(np.uint8)
    img_out  = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    return file_name.split('.')[0], img_out


def io_producer(scene_batch, save_folder, args, gpu_queues, upload_queue):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    num_gpus = len(gpu_queues)
    shape    = (args.resolution, args.resolution)
    h_out, w_out = shape
    frame_idx = 0

    for directory in tqdm(scene_batch, desc="  Processing scenes", ascii=True):
        img_folder = os.path.join(save_folder, directory, "iphone", "rgb")
        if not os.path.exists(img_folder):
            print(f"[Warning] No rgb folder for '{directory}', skipping.", flush=True)
            continue

        directory_data_list = sorted(os.listdir(img_folder), key=natural_sort_key)

        # Resume: skip already-processed frames
        last_done = get_last_completed_stem(save_folder, directory, args.encoder)
        if last_done is not None:
            stems = [f.split('.')[0] for f in directory_data_list]
            if last_done in stems:
                resume_idx = stems.index(last_done) + 1
                print(f"[Resume] '{directory}': skipping {resume_idx} frames", flush=True)
                directory_data_list = directory_data_list[resume_idx:]

        if not directory_data_list:
            upload_queue.put((os.path.join(save_folder, directory), directory))
            continue

        # Write resolution file once
        res_file = os.path.join(save_folder, "resolution.txt")
        if not os.path.exists(res_file):
            with open(res_file, "w") as f:
                f.write(f"{w_out} {h_out}\n")

        print(f"[Producer] '{directory}': {len(directory_data_list)} frames", flush=True)

        all_files = list(directory_data_list)
        with ProcessPoolExecutor(max_workers=LOAD_WORKERS) as pool:
            for batch_start in range(0, len(all_files), FRAME_BATCH):
                batch   = all_files[batch_start: batch_start + FRAME_BATCH]
                futures = {pool.submit(load_frame_task, (fn, img_folder, shape)): fn for fn in batch}
                for future in tqdm(as_completed(futures), total=len(batch),
                                   desc=f"  {directory} [{batch_start}+]", ascii=True):
                    stem, img_out = future.result()
                    if img_out is None:
                        continue
                    # Round-robin across GPUs — use put() with no timeout;
                    # queue is large enough (workers_per_gpu * 64) that this
                    # should never block for long.
                    gpu_queues[frame_idx % num_gpus].put(
                        (directory, stem, img_out, w_out, h_out)
                    )
                    frame_idx += 1

        upload_queue.put((os.path.join(save_folder, directory), directory))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def make_workers(args, gpu_queues, save_folder):
    workers = []
    for gpu_id in range(args.num_gpus):
        for w in range(args.workers_per_gpu):
            worker_id = gpu_id * args.workers_per_gpu + w
            p = Process(
                target=gpu_worker,
                args=(worker_id, gpu_id, gpu_queues[gpu_id],
                      args.sam_ckpt_path, args.encoder, args.empty_bg, save_folder),
            )
            p.start()
            workers.append(p)
    return workers


def stop_workers(workers, gpu_queues):
    for q in gpu_queues:
        q.put(SENTINEL)
    for p in workers:
        p.join()


if __name__ == '__main__':
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution',      type=int, default=256)
    parser.add_argument('--sam_ckpt_path',   type=str,
                        default="/dss/dsshome1/03/di38wok/Projects/Preproccessing/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder',         type=str, default="clip")
    parser.add_argument('--empty_bg',        action='store_true', default=False)
    parser.add_argument('--num_gpus',        type=int, default=1)
    parser.add_argument('--workers_per_gpu', type=int, default=1)
    args = parser.parse_args()

    save_folder = "/tmp/dataset/frames"
    os.makedirs(save_folder, exist_ok=True)

    # ── Build scene list ──────────────────────────────────────────────────
    api = HfApi(token=HF_TOKEN)
    print("Fetching scene list from HuggingFace...", flush=True)
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    data_list = sorted({
        path.split("/")[0]
        for path in all_files
        if "/" in path
    }, reverse=True)
    print(f"Found {len(data_list)} scenes on HuggingFace: {REPO_ID}", flush=True)

    # ── Filter already-processed scenes ──────────────────────────────────
    print("Checking which scenes are already processed...", flush=True)
    todo_list = []
    for directory in tqdm(data_list, desc="Checking HF", ascii=True):
        if not scene_already_processed(api, REPO_ID, directory):
            todo_list.append(directory)
    print(f"{len(todo_list)} scenes to process out of {len(data_list)}", flush=True)

    if not todo_list:
        print("Nothing to do. All scenes already processed.", flush=True)
        exit(0)

    total_workers = args.num_gpus * args.workers_per_gpu
    queue_maxsize = args.workers_per_gpu * 64
    print(f"Launching {args.num_gpus} GPU(s) x {args.workers_per_gpu} worker(s) = {total_workers} workers", flush=True)

    gpu_queues = [Queue(maxsize=queue_maxsize) for _ in range(args.num_gpus)]

    upload_queue: thread_queue.Queue = thread_queue.Queue()
    uploader = threading.Thread(target=hf_upload_thread, args=(upload_queue,), daemon=True)
    uploader.start()

    # ── Batch loop: download N scenes → process → repeat ─────────────────
    n_batches = (len(todo_list) + BATCH_SCENES - 1) // BATCH_SCENES
    for batch_idx, batch_start in enumerate(range(0, len(todo_list), BATCH_SCENES)):
        batch = todo_list[batch_start: batch_start + BATCH_SCENES]
        print(f"\n=== Batch {batch_idx+1}/{n_batches}: {len(batch)} scenes ===", flush=True)

        # 1. Download
        print("Downloading...", flush=True)
        for directory in tqdm(batch, desc="  Downloading", ascii=True):
            download_scene(api, REPO_ID, directory, save_folder)
        print("Download complete.", flush=True)

        # 2. Start workers fresh for this batch
        workers = make_workers(args, gpu_queues, save_folder)

        # 3. Feed frames to workers
        io_producer(batch, save_folder, args, gpu_queues, upload_queue)

        # 4. Drain workers and wait
        stop_workers(workers, gpu_queues)

        # 5. Recreate queues for next batch (drained queues can't be reused safely)
        gpu_queues = [Queue(maxsize=queue_maxsize) for _ in range(args.num_gpus)]

        print(f"Batch {batch_idx+1} complete.", flush=True)

    # ── Wait for all uploads to finish ────────────────────────────────────
    upload_queue.put(None)
    uploader.join()
    print("All done.", flush=True)
