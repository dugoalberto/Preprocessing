import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torch.utils._contextlib import F


def rescale(
        image: torch.Tensor,  # [3, h, w]  float32 in [0, 1]
        shape: tuple[int, int],
) -> torch.Tensor:  # [3, h_out, w_out]
    h, w = shape
    img_u8 = (image * 255).clamp(0, 255).to(torch.uint8)
    img_np = rearrange(img_u8, "c h w -> h w c").cpu().numpy()
    img_pil = Image.fromarray(img_np).resize((w, h), Image.BILINEAR)
    img_out = torch.tensor(np.array(img_pil) / 255.0, dtype=image.dtype, device=image.device)
    return rearrange(img_out, "h w c -> c h w")


def center_crop(
        image: torch.Tensor,  # [c, h_in, w_in]
        intrinsics: torch.Tensor,  # [3, 3]   normalised (cx/w, cy/h in [0,1])
        shape: tuple[int, int]
) -> tuple:
    h_in, w_in = image.shape[-2:]
    h_out, w_out = shape

    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # crop image
    image = image[..., row: row + h_out, col: col + w_out]

    # update intrinsics (normalised convention: fx=fx_px/w, cx=cx_px/w)
    intr = intrinsics.clone()
    intr[0, 0] *= w_in / w_out  # fx
    intr[1, 1] *= h_in / h_out  # fy
    intr[0, 2] = (intr[0, 2] * w_in - col) / w_out  # cx
    intr[1, 2] = (intr[1, 2] * h_in - row) / h_out  # cy

    return image, intr


def rescale_and_crop(
        image: torch.Tensor,  # [3, h_in, w_in]  float32 [0,1]
        intrinsics: torch.Tensor,  # [3, 3]
        shape: tuple[int, int],
) -> tuple:
    h_in, w_in = image.shape[-2:]
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in, \
        f"Target {shape} must be <= input ({h_in},{w_in})"

    # isotropic scale: shorter output dim just meets the target
    scale = max(h_out / h_in, w_out / w_in)
    h_sc = round(h_in * scale)
    w_sc = round(w_in * scale)
    assert h_sc == h_out or w_sc == w_out, \
        f"Unexpected scaled size ({h_sc},{w_sc}) for target {shape}"

    # rescale image
    image = rescale(image, (h_sc, w_sc))

    # rescale intrinsics for the isotropic scale step
    intr = intrinsics.clone()
    intr[0, 0] *= w_sc / w_in  # fx
    intr[1, 1] *= h_sc / h_in  # fy
    intr[0, 2] *= w_sc / w_in  # cx
    intr[1, 2] *= h_sc / h_in  # cy

    return center_crop(image, intr, shape)
