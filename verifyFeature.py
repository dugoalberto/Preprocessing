"""
Downloads image + CLIP features for a single frame from HuggingFace,
processes the seg_map → feature_map, and saves visualizations.

Dataset : dugoalberto/Scannet_Clip
Scene   : 00777c41d4
Frame   : frame_000000
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
REPO_ID   = "dugoalberto/Scannet_Clip"
SCENE     = "00a231a370"
FRAME     = "frame_000016"
H, W      = 256, 256                     # target spatial resolution
OUT_DIR   = f"/mnt/home/albertodugo/Projects/Preproccessing/Datasets/{SCENE}"

# HF paths
IMG_PATH  = f"{SCENE}/iphone/rgb/{FRAME}.jpg"
FEAT_PATH = f"{SCENE}/features/clip/{FRAME}_feats.npy"
SEG_PATH  = f"{SCENE}/features/clip/{FRAME}_seg_map.npy"


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def download(repo_id: str, path: str) -> str:
    """Download a single file from an HF dataset repo and return its local path."""
    return hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")


def resize_center_crop(seg_map_hw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rescale + center-crop a 2-D segmentation map to (H, W),
    exactly as done in load_target_feature.
    """
    h_in, w_in = seg_map_hw.shape
    scale_factor = max(H / h_in, W / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    seg = seg_map_hw.unsqueeze(0).unsqueeze(0).float()          # [1,1,h,w]
    seg = F.interpolate(seg, size=(h_scaled, w_scaled), mode="nearest")

    row = (h_scaled - H) // 2
    col = (w_scaled - W) // 2
    seg = seg[..., row: row + H, col: col + W]                  # [1,1,H,W]
    return seg.squeeze(0).squeeze(0).long()                     # [H,W]


def get_feature_map(seg_map: torch.Tensor, features: torch.Tensor):
    """
    Map per-segment CLIP features onto a dense pixel grid.
    seg_map  : [H, W]  long  – segment indices
    features : [N, C]  float – one feature vector per segment
    Returns:
        feature_map : [C, H, W]
        valid_mask  : [H, W]   bool
    """
    H, W = seg_map.shape
    N, C = features.shape

    # flat index into features; -1 → invalid
    flat = seg_map.reshape(-1)                                  # [H*W]
    valid = (flat >= 0) & (flat < N)

    feature_map = torch.zeros(H * W, C, device=features.device)
    feature_map[valid] = features[flat[valid]]
    feature_map = feature_map.reshape(H, W, C).permute(2, 0, 1)  # [C,H,W]
    valid_mask  = valid.reshape(H, W)

    return feature_map, valid_mask


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # ── 1. Download from HF ──────────────────
    print("Downloading image …")
    img_local  = download(REPO_ID, IMG_PATH)
    print("Downloading features …")
    feat_local = download(REPO_ID, FEAT_PATH)
    print("Downloading seg map …")
    seg_local  = download(REPO_ID, SEG_PATH)

    # ── 2. Load image (original + resized) ───
    img_orig = Image.open(img_local).convert("RGB")
    img_orig_np = np.array(img_orig)                            # [H_orig, W_orig, 3]

    out_img_orig = os.path.join(OUT_DIR, f"{FRAME}_original.jpg")
    img_orig.save(out_img_orig)
    print(f"Saved original image → {out_img_orig}")

    img = img_orig.resize((W, H), Image.BILINEAR)
    img_np = np.array(img)                                      # [H,W,3]  uint8

    out_img = os.path.join(OUT_DIR, f"{FRAME}.jpg")
    img.save(out_img)
    print(f"Saved resized image → {out_img}")

    # ── 3. Load features & seg map ───────────
    features = torch.from_numpy(
        np.load(feat_local, allow_pickle=True)
    ).float()                                                   # [N, C]

    masks   = np.load(seg_local, allow_pickle=True).item()
    seg_map = torch.from_numpy(masks["l"])                     # [h, w]  (original res)

    print(f"Features shape : {features.shape}")
    print(f"Seg map shape  : {seg_map.shape}  (before resize)")

    # ── 4. Resize seg map ────────────────────
    seg_map = resize_center_crop(seg_map, H, W)                # [H, W]
    print(f"Seg map shape  : {seg_map.shape}  (after resize)")

    # ── 5. Build dense feature map ───────────
    feature_map, valid_mask = get_feature_map(seg_map, features)
    print(f"Feature map shape : {feature_map.shape}")          # [C, H, W]
    print(f"Valid pixels      : {valid_mask.sum().item()} / {H*W}")

    # ── 6. Save seg map visualisation ────────
    seg_np = seg_map.numpy()                                    # [H, W]
    n_segs = int(seg_np.max()) + 1
    cmap   = cm.get_cmap("tab20", n_segs)

    seg_rgba = cmap(seg_np % 20)                               # [H,W,4]  float 0-1
    seg_rgb  = (seg_rgba[..., :3] * 255).astype(np.uint8)

    out_seg = os.path.join(OUT_DIR, f"{FRAME}_seg_map.png")
    Image.fromarray(seg_rgb).save(out_seg)
    print(f"Saved seg map visualisation → {out_seg}")

    # ── 7. Save feature map (PCA → RGB) ──────
    feat_np = feature_map.permute(1, 2, 0).numpy()             # [H, W, C]
    valid   = valid_mask.numpy()                                # [H, W]

    # PCA to 3 dims for visualisation
    pixels = feat_np[valid]                                     # [M, C]
    pixels -= pixels.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(pixels, full_matrices=False)
    proj = (pixels @ Vt[:3].T)                                 # [M, 3]

    # normalise to [0,1]
    proj -= proj.min(0, keepdims=True)
    proj /= proj.max(0, keepdims=True) + 1e-8

    feat_rgb = np.zeros((H, W, 3), dtype=np.float32)
    feat_rgb[valid] = proj
    feat_rgb_uint8 = (feat_rgb * 255).astype(np.uint8)

    out_feat = os.path.join(OUT_DIR, f"{FRAME}_feature_map_pca.png")
    Image.fromarray(feat_rgb_uint8).save(out_feat)
    print(f"Saved feature map PCA visualisation → {out_feat}")

    # ── 8. Combined figure ───────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Image (original)", "Image (256×256)", "Seg Map", "Feature Map (PCA)"]
    imgs   = [img_orig_np, img_np, seg_rgb, feat_rgb_uint8]

    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(im)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"{FRAME}_overview.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview figure → {out_fig}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()