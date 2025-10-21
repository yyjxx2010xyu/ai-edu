import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


def list_images(root):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    files = sorted(list(set(files)))
    return files


def pil_to_tensor(im: Image.Image):
    arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (0, 1, 2))
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


class TextureTilesDataset(Dataset):
    def __init__(self, root_dir, tile_size=64):
        super().__init__()
        self.files = list_images(root_dir)
        self.tile_size = tile_size
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def _split_into_tiles(self, img_t: torch.Tensor):
        # img_t: (C,H,W) in [0,1]
        C, H, W = img_t.shape
        ts = self.tile_size
        tiles = []
        # If not divisible, pad using reflect to cover at least one tile
        pad_h = (ts - H % ts) % ts
        pad_w = (ts - W % ts) % ts
        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            C, H, W = img_t.shape
        for y in range(0, H, ts):
            for x in range(0, W, ts):
                tiles.append(img_t[:, y:y+ts, x:x+ts])
        # Return one random tile per image to keep dataset small
        idx = np.random.randint(0, len(tiles))
        return tiles[idx]

    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as im:
            t = pil_to_tensor(im)
        tile = self._split_into_tiles(t)
        return tile
