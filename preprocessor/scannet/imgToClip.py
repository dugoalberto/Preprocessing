# clip_feature_extractor.py
import torch
import numpy as np
import torchvision.transforms.functional as TF

from Preproccessing.preprocessor.utils.CLIP import OpenCLIPNetwork


def extract_clip_feature_map(
    img: np.ndarray,           # H x W x 3, uint8, RGB
    pix_obj_ids: np.ndarray,   # H x W, int, -1 = background
    clip_model: OpenCLIPNetwork,
    bbox_expand_factor: float = 1.2,
    mask_background: bool = False,   # set True for no-bg crops
) -> torch.Tensor:
    """
    Returns a [512, H, W] float32 tensor of CLIP features per pixel.
    Pixels with pix_obj_ids == -1 get a zero vector.
    """
    H, W = pix_obj_ids.shape
    feature_map = torch.zeros(clip_model.clip_n_dims, H, W, dtype=torch.float32)

    obj_ids = np.unique(pix_obj_ids)
    obj_ids = obj_ids[(obj_ids != -1) & (obj_ids != 0)]   # skip bg / unlabelled

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]

    crops, valid_ids = [], []

    for obj_id in obj_ids:
        mask = pix_obj_ids == obj_id
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue

        # bbox with expand
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        rh, rw = r1 - r0, c1 - c0
        pad_r = int(rh * (bbox_expand_factor - 1) / 2)
        pad_c = int(rw * (bbox_expand_factor - 1) / 2)
        r0 = max(0, r0 - pad_r);  r1 = min(H - 1, r1 + pad_r)
        c0 = max(0, c0 - pad_c);  c1 = min(W - 1, c1 + pad_c)

        crop = img_tensor[:, r0:r1+1, c0:c1+1].clone()  # [3, h, w]

        if mask_background:
            obj_mask = torch.from_numpy(mask[r0:r1+1, c0:c1+1])
            crop[:, ~obj_mask] = 0.0

        crops.append(crop)
        valid_ids.append(obj_id)

    if not crops:
        return feature_map

    # Batch-encode: resize all crops to 224x224 first
    resized = torch.stack([
        TF.resize(c, [224, 224], antialias=True) for c in crops
    ])  # [N, 3, 224, 224]

    with torch.no_grad():
        # clip_model.process expects already-resized tensors
        normed = clip_model.process[1](resized).cuda()   # only the Normalize step
        feats = clip_model.model.encode_image(normed.half())   # [N, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.float().cpu()

    # Splat back onto pixel grid
    for feat, obj_id in zip(feats, valid_ids):
        mask = torch.from_numpy(pix_obj_ids == obj_id)   # [H, W]
        feature_map[:, mask] = feat.unsqueeze(1)          # broadcast [512, npix]

    return feature_map   # [512, H, W]