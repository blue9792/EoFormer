import glob
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from monai.data import CacheDataset
from monai.transforms import Resize, MapTransform


class NpyBraTSDataset(Dataset):
    REGION_TYPES = ("WT", "TC", "ET")

    def __init__(
        self,
        npy_root: str,
        num_frames: int = 16,
        min_tumor_slices: int = 4,
        reverse_time_prob: float = 0.0,
        seed: int = 123,
        cache_npy_in_ram: bool = False,
        target_hw: Tuple[int, int] = (256, 256),
    ) -> None:
        self.npy_root = npy_root
        self.num_frames = num_frames
        self.min_tumor_slices = min_tumor_slices
        self.reverse_time_prob = reverse_time_prob
        self.seed = seed
        self.cache_npy_in_ram = cache_npy_in_ram
        self.target_hw = target_hw
        self.epoch = 0

        all_x_npy = sorted(glob.glob(os.path.join(npy_root, "*_x.npy")))
        if len(all_x_npy) == 0:
            raise RuntimeError(f"No npy files found under: {npy_root}")

        self.patient_files = []
        for x_path in all_x_npy:
            seg_path = x_path.replace("_x.npy", "_seg.npy")
            if not os.path.exists(seg_path):
                continue
            seg = np.load(seg_path, mmap_mode="r" if not self.cache_npy_in_ram else None)
            tumor_per_slice = (seg > 0).any(axis=(0, 1))
            if int(tumor_per_slice.sum()) < self.min_tumor_slices:
                continue
            self.patient_files.append(x_path)

        if len(self.patient_files) == 0:
            raise RuntimeError(
                f"No valid npy pairs with >= {self.min_tumor_slices} tumor slices under: {npy_root}"
            )

        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.patient_files)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _make_region_mask(seg_slice: np.ndarray, region: str) -> np.ndarray:
        if region == "WT":
            mask = seg_slice > 0
        elif region == "TC":
            mask = (seg_slice == 1) | (seg_slice == 4)
        elif region == "ET":
            mask = seg_slice == 4
        else:
            raise ValueError(region)
        return mask.astype(np.uint8)

    def _load_npy(self, x_path: str) -> Dict[str, np.ndarray]:
        pid = os.path.basename(x_path).replace("_x.npy", "")

        if self.cache_npy_in_ram and pid in self._cache:
            return self._cache[pid]

        seg_path = x_path.replace("_x.npy", "_seg.npy")
        x = np.load(x_path, mmap_mode="r" if not self.cache_npy_in_ram else None)
        seg = np.load(seg_path, mmap_mode="r" if not self.cache_npy_in_ram else None)

        if self.cache_npy_in_ram:
            out = {"x": x.copy(), "seg": seg.copy()}
            self._cache[pid] = out
            return out

        return {"x": x, "seg": seg}

    def _sample_indices(self, tumor_per_slice: np.ndarray, rng: np.random.RandomState) -> List[int]:
        depth = tumor_per_slice.shape[0]
        num_frames = min(self.num_frames, depth)
        valid_starts = np.arange(0, depth - num_frames + 1)
        if len(valid_starts) == 0:
            return list(range(depth))

        tumor_counts = np.convolve(
            tumor_per_slice.astype(np.int32),
            np.ones(num_frames, dtype=np.int32),
            mode="valid",
        )
        if self.min_tumor_slices > 0:
            candidates = valid_starts[tumor_counts >= self.min_tumor_slices]
            if len(candidates) == 0:
                candidates = valid_starts
        else:
            candidates = valid_starts

        start = int(rng.choice(candidates))
        indices = list(range(start, start + num_frames))
        if self.reverse_time_prob > 0 and rng.rand() < self.reverse_time_prob:
            indices = indices[::-1]
        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx + self.epoch * 1000003)
        for _ in range(10):
            x_path = self.patient_files[idx]
            arrays = self._load_npy(x_path)
            x = arrays["x"]
            seg = arrays["seg"]

            tumor_per_slice = (seg > 0).any(axis=(0, 1))
            z_inds = self._sample_indices(tumor_per_slice, rng)
            if int(tumor_per_slice[z_inds].sum()) >= self.min_tumor_slices:
                break
            idx = int(rng.randint(0, len(self.patient_files)))
        else:
            raise RuntimeError("Failed to sample a valid clip with tumor slices.")

        frames = np.stack([x[:, :, :, z] for z in z_inds], axis=0).astype(np.float32)
        all_masks = []
        for z in z_inds:
            slice_seg = seg[:, :, z]
            m_wt = self._make_region_mask(slice_seg, "WT")
            m_tc = self._make_region_mask(slice_seg, "TC")
            m_et = self._make_region_mask(slice_seg, "ET")
            all_masks.append(np.stack([m_wt, m_tc, m_et], axis=0))

        frames_t = torch.from_numpy(frames)
        masks_t = torch.from_numpy(np.stack(all_masks, axis=0)).float()

        target_hw = self.target_hw
        if target_hw is not None:
            import torch.nn.functional as F

            frames_t = F.interpolate(
                frames_t,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            masks_t = F.interpolate(
                masks_t,
                size=target_hw,
                mode="nearest",
            )

        frames_t = frames_t.permute(1, 0, 2, 3).contiguous()
        masks_t = masks_t.permute(1, 0, 2, 3).contiguous()

        return {
            "image": frames_t,
            "label": masks_t,
        }

def split_data(datalist, basedir):

    with open(datalist) as f:
        json_data = json.load(f)

    train_data = json_data['train']
    valid_data = json_data['valid']
    test_data = json_data['test']
    
    train = []
    valid = []
    test = []
    for d in train_data:
        for k, v in d.items():  
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        train.append(d)

    for d in valid_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        valid.append(d)
    
    for d in test_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        test.append(d)
    return train, valid, test


def get_dataset_brats(data_path: str, 
                  json_file: str, 
                  transform=None,
                 )-> Tuple[Dataset, Dataset]:
    
    train_list, valid_list, test_list = split_data(json_file, data_path)
    
    train_set = CacheDataset(
                            data = train_list,
                            cache_rate=0.0,
                            transform = transform['train']
                            )
    valid_set = CacheDataset(
                            data = valid_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    test_set = CacheDataset(
                            data = test_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    
    return train_set, valid_set, test_set, train_list, valid_list, test_list


def get_dataset_npy(
    train_dir: str,
    val_dir: str,
    num_frames: int = 16,
    min_tumor_slices: int = 4,
    reverse_time_prob: float = 0.0,
    seed: int = 123,
    cache_npy_in_ram: bool = False,
    target_hw: Tuple[int, int] = (256, 256),
) -> Tuple[Dataset, Dataset]:
    train_set = NpyBraTSDataset(
        npy_root=train_dir,
        num_frames=num_frames,
        min_tumor_slices=min_tumor_slices,
        reverse_time_prob=reverse_time_prob,
        seed=seed,
        cache_npy_in_ram=cache_npy_in_ram,
        target_hw=target_hw,
    )
    valid_set = NpyBraTSDataset(
        npy_root=val_dir,
        num_frames=num_frames,
        min_tumor_slices=min_tumor_slices,
        reverse_time_prob=0.0,
        seed=seed,
        cache_npy_in_ram=cache_npy_in_ram,
        target_hw=target_hw,
    )
    return train_set, valid_set
