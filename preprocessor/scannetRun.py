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
SENTINEL = None

REPO_ID = "dugoalberto/Scannet_Clip"
HF_TOKEN = "-"

def scene_already_processed(api: HfApi, repo_id: str, scene_name: str) -> bool:
    """Check if both 'features' and 'SAM' dirs exist for this scene on HF."""
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
    """
    Downloads only the iphone/rgb frames for a scene into local_dir/scene_name.
    """
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,          # <-- save_folder, not save_folder/scene_name
        allow_patterns=[f"{scene_name}/iphone/rgb/*"],
        token=HF_TOKEN,
    )
    return os.path.join(local_dir, scene_name)
# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]


def get_last_completed_stem(save_folder: str, directory: str, encoder: str) -> str | None:
    """
    Returns the stem of the last successfully saved frame, for resume logic.
    """
    feat_dir = os.path.join(save_folder, directory, "features", encoder)
    if not os.path.exists(feat_dir):
        return None
    npy_files = [f for f in os.listdir(feat_dir) if f.endswith("_seg_map.npy")]
    if not npy_files:
        return None
    npy_files.sort(key=natural_sort_key)
    return npy_files[-1][: -len("_seg_map.npy")]


# ─────────────────────────────────────────────────────────────────────────────
# Saver thread  (runs inside each GPU worker process)
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
            print(f"[Saver] ERROR saving {args[2]}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace upload thread  (runs in the MAIN process, async per scene)
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

            # ── Step 5: clean up local scene to free disk ─────────────────
            shutil.rmtree(scene_local_path, ignore_errors=True)
            print(f"[HF] Cleaned up local: {scene_local_path}", flush=True)
        except Exception as e:
            print(f"[HF] Upload ERROR for {scene_repo_path}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPU worker process
# One process per (gpu_id, worker_slot). Each process owns its own models.
# All workers for the same GPU share the same input Queue.
# ─────────────────────────────────────────────────────────────────────────────
def gpu_worker(worker_id: int, device_id: int, in_queue: Queue,
               sam_ckpt_path: str, encoder: str, empty_bg: bool, save_folder: str):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam_processor = SAMProcessor(sam_ckpt_path=sam_ckpt_path, device=device)
    feature_extractors: dict[str, FeatureExtractor] = {}

    # Background saver so GPU never waits on disk I/O
    save_queue: thread_queue.Queue = thread_queue.Queue(maxsize=8)
    saver = threading.Thread(target=saver_thread, args=(save_queue,), daemon=True)
    saver.start()

    print(f"[Worker {worker_id}] Ready on {device}", flush=True)

    while True:
        item = in_queue.get()
        if item is SENTINEL:
            # Re-enqueue sentinel so other workers on the same queue also stop
            in_queue.put(SENTINEL)
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
            sam_result = sam_processor.process_images_nosave(
                image, file_stem, f"{save_folder}/{directory}", empty_bg=empty_bg
            )
            feat_result = feature_extractors[directory].create_features_nosave(
                sam_result, method=encoder, level='l'
            )

        save_queue.put((do_save, (save_folder, directory, file_stem, encoder,
                                  sam_result, feat_result)))


# ─────────────────────────────────────────────────────────────────────────────
# I/O producer  (main process, single thread)
# One Queue per GPU (shared by all workers on that GPU).
# Round-robin distributes frames across GPUs.
# HF upload is fire-and-forget via upload_queue.
# ─────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor

# signature
def io_producer(data_list, save_folder, args, gpu_queues, upload_queue):
    api = HfApi(token=HF_TOKEN)
    num_gpus = len(gpu_queues)
    shape = (args.resolution, args.resolution)
    h_out, w_out = shape
    frame_idx = 0

    for directory in tqdm(data_list, desc="Scenes", ascii=True):

        # ── Step 1: already processed on HF? ─────────────────────────────
        if scene_already_processed(api, REPO_ID, directory):
            print(f"[HF] '{directory}' already processed, skipping.")
            continue

        download_scene(api, REPO_ID, directory, save_folder)

        img_folder = os.path.join(save_folder, directory, "iphone", "rgb")
        if not os.path.exists(img_folder):
            print(f"[Warning] No rgb folder found for '{directory}', skipping.")
            continue

        directory_data_list = sorted(os.listdir(img_folder), key=natural_sort_key)

        # ── Resume (partial local processing) ────────────────────────────
        last_done = get_last_completed_stem(save_folder, directory, args.encoder)
        if last_done is not None:
            stems = [f.split('.')[0] for f in directory_data_list]
            if last_done in stems:
                resume_idx = stems.index(last_done) + 1
                print(f"[Resume] '{directory}': skipping {resume_idx} frames")
                directory_data_list = directory_data_list[resume_idx:]

        if not directory_data_list:
            # Already fully processed locally, just upload
            upload_queue.put((os.path.join(save_folder, directory), directory))
            continue

        # ── Intrinsics ────────────────────────────────────────────────────
        first_frame = cv2.imread(os.path.join(img_folder, directory_data_list[0]))
        res_file = os.path.join(save_folder, "resolution.txt")
        if not os.path.exists(res_file):
            with open(res_file, "w") as f:
                f.write(f"{w_out} {h_out}\n")

        # ── Step 3: enqueue frames for GPU processing ─────────────────────
        def load_frame(file_name):
            bgr = cv2.imread(os.path.join(img_folder, file_name))
            if bgr is None:
                return None, None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = rearrange(torch.from_numpy(rgb / 255.0).float(), "h w c -> c h w")
            img_crop = rescale_and_crop(img_t, shape)
            img_out = (rearrange(img_crop, "c h w -> h w c").numpy() * 255).astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            return file_name.split('.')[0], img_out

        print(f"[Producer] '{directory}': {len(directory_data_list)} frames")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        LOAD_WORKERS = 4  # fewer parallel readers
        BATCH_SIZE = 32  # how many frames to hold in RAM at once

        with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
            all_files = list(directory_data_list)
            for batch_start in range(0, len(all_files), BATCH_SIZE):
                batch = all_files[batch_start: batch_start + BATCH_SIZE]
                futures = {pool.submit(load_frame, fn): fn for fn in batch}
                for future in tqdm(as_completed(futures), total=len(batch),
                                   desc=f"  {directory} [{batch_start}+]", ascii=True):
                    stem, img_out = future.result()
                    if img_out is None:
                        continue
                    gpu_queues[frame_idx % num_gpus].put(
                        (directory, stem, img_out, w_out, h_out)
                    )
                    frame_idx += 1        # ── Steps 4 & 5: upload features+SAM back, then clean up ──────────
        upload_queue.put((os.path.join(save_folder, directory), directory))

    for q in gpu_queues:
        q.put(SENTINEL)
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution',      type=int, default=256)
    parser.add_argument('--sam_ckpt_path',   type=str,
                        default="/mnt/home/albertodugo/Projects/Preproccessing/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder',         type=str, default="clip")
    parser.add_argument('--empty_bg',        action='store_true', default=False)
    parser.add_argument('--num_gpus',        type=int, default=1)
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    args = parser.parse_args()

    save_folder = "/tmp/dataset/frames"
    os.makedirs(save_folder, exist_ok=True)

    # ── Fetch scene list directly from HuggingFace ───────────────────────
    api = HfApi(token=HF_TOKEN)
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    data_list = sorted({
        path.split("/")[0]
        for path in all_files
        if "/" in path  # skip any root-level files
    })
    print(f"Found {len(data_list)} scenes on HuggingFace: {REPO_ID}")

    total_workers = args.num_gpus * args.workers_per_gpu
    print(f"Launching {args.num_gpus} GPU(s) × {args.workers_per_gpu} worker(s) = {total_workers} workers total")

    queue_maxsize = args.workers_per_gpu * 16
    gpu_queues = [Queue(maxsize=queue_maxsize) for _ in range(args.num_gpus)]

    # Async HuggingFace uploader running in the main process
    upload_queue: thread_queue.Queue = thread_queue.Queue()
    uploader = threading.Thread(target=hf_upload_thread, args=(upload_queue,), daemon=True)
    uploader.start()

    # Spawn GPU workers
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

    # Run producer (blocks until all frames are enqueued)
    # in __main__
    io_producer(data_list, save_folder, args, gpu_queues, upload_queue)

    for p in workers:
        p.join()

    upload_queue.put(None)
    uploader.join()

    print("All done.")