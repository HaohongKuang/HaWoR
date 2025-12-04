import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate

from hawor.configs import CACHE_DIR_HAWOR
from hawor.utils.pylogger import get_pylogger
from lib.models.mano_wrapper import MANO
from lib.utils.imutils import (
    crop,
    crop_j2d,
    est_intrinsics,
    flip_img,
    flip_kp,
    flip_pose,
    get_normalization,
    rot_aa,
    transform_pts,
)

log = get_pylogger(__name__)


def _load_annotation_file(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[-1]
    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return data["samples"] if isinstance(data, dict) and "samples" in data else data
    if ext in {".npz", ".npy"}:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            return data["samples"].tolist()
        return data.tolist()
    raise ValueError(f"Unsupported annotation format: {ext}")


def _prepare_mano(cfg) -> Optional[MANO]:
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    if not os.path.exists(mano_cfg["model_path"]):
        log.warning(
            "MANO model path %s was not found. Ground-truth joints will be read from annotations only.",
            mano_cfg["model_path"],
        )
        return None
    return MANO(**mano_cfg)


def _stack_dict(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    stacked = {}
    for key in batch[0]:
        stacked[key] = torch.stack([item[key] for item in batch], dim=0)
    return stacked


class HaWoRDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        cfg,
        is_train: bool = True,
        image_root: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.seq_len = cfg.DATASETS.get("SEQ_LEN", cfg.TRAIN.SEQ_LEN)
        self.image_root = image_root or cfg.DATASETS.IMAGE_ROOT or CACHE_DIR_HAWOR
        self.crop_size = cfg.MODEL.IMAGE_SIZE
        self.aug_cfg = cfg.DATASETS.CONFIG
        self.normalize = get_normalization()
        self.mano_layer = _prepare_mano(cfg)

        raw_samples = _load_annotation_file(annotation_path)
        sequences: Dict[str, List[Dict]] = {}
        for sample in raw_samples:
            seq_id = sample.get("sequence_id", "default")
            sequences.setdefault(seq_id, []).append(sample)

        self.sequences: Dict[str, List[Dict]] = {}
        for seq_id, frames in sequences.items():
            frames = sorted(frames, key=lambda x: x.get("frame_id", 0))
            self.sequences[seq_id] = frames

        self.index: List[Tuple[str, int]] = []
        for seq_id, frames in self.sequences.items():
            if len(frames) < self.seq_len:
                log.warning(
                    "Sequence %s shorter (%d) than seq_len (%d); skipping.",
                    seq_id,
                    len(frames),
                    self.seq_len,
                )
                continue
            for start in range(0, len(frames) - self.seq_len + 1):
                self.index.append((seq_id, start))

        if len(self.index) == 0:
            raise ValueError(
                "No sequences available for training/validation. Check your annotations and seq_len settings."
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_id, start = self.index[idx]
        frames = self.sequences[seq_id][start : start + self.seq_len]

        aug_params = self._sample_augmentation()
        processed_frames = [self._process_frame(sample, aug_params) for sample in frames]

        batch = _stack_dict(processed_frames)
        return batch

    def _sample_augmentation(self) -> Dict:
        if not self.is_train:
            return {
                "scale_noise": 1.0,
                "rot": 0.0,
                "trans": np.zeros(2, dtype=np.float32),
                "flip": False,
                "extreme_crop": False,
            }

        scale_noise = np.clip(
            np.random.randn() * self.aug_cfg.SCALE_FACTOR + 1,
            1 - self.aug_cfg.SCALE_FACTOR,
            1 + self.aug_cfg.SCALE_FACTOR,
        )
        rot = 0.0
        if np.random.rand() <= self.aug_cfg.ROT_AUG_RATE:
            rot = np.clip(
                np.random.randn() * self.aug_cfg.ROT_FACTOR,
                -2 * self.aug_cfg.ROT_FACTOR,
                2 * self.aug_cfg.ROT_FACTOR,
            )
        trans = np.zeros(2, dtype=np.float32)
        if np.random.rand() <= self.aug_cfg.TRANS_AUG_RATE:
            trans = np.random.randn(2).astype(np.float32) * self.aug_cfg.TRANS_FACTOR

        flip = False
        if self.aug_cfg.DO_FLIP and np.random.rand() <= self.aug_cfg.FLIP_AUG_RATE:
            flip = True

        extreme_crop = np.random.rand() <= self.aug_cfg.EXTREME_CROP_AUG_RATE
        return {
            "scale_noise": scale_noise,
            "rot": rot,
            "trans": trans,
            "flip": flip,
            "extreme_crop": extreme_crop,
        }

    def _resolve_paths(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.image_root, path)

    def _process_frame(self, sample: Dict, aug_params: Dict) -> Dict[str, torch.Tensor]:
        img_path = self._resolve_paths(sample["img_path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Could not find image at {img_path}")

        img = cv2.imread(img_path)[:, :, ::-1]
        img_h, img_w = img.shape[:2]

        if "center" in sample and "scale" in sample:
            center = np.array(sample["center"], dtype=np.float32)
            scale = float(sample["scale"])
        elif "bbox" in sample:
            x1, y1, x2, y2 = np.array(sample["bbox"], dtype=np.float32)
            w, h = x2 - x1, y2 - y1
            center = np.array([x1 + w / 2.0, y1 + h / 2.0], dtype=np.float32)
            scale = max(w, h) / 200.0
        else:
            raise KeyError("Each annotation must include either center/scale or bbox.")

        gt_j2d = np.array(sample["gt_cam_j2d"], dtype=np.float32)
        gt_pose = np.array(sample["gt_cam_full_pose"], dtype=np.float32)
        betas = np.array(sample.get("gt_cam_betas", sample.get("betas", np.zeros(10))), dtype=np.float32)
        gt_j3d = sample.get("gt_j3d_wo_trans")
        if gt_j3d is not None:
            gt_j3d = np.array(gt_j3d, dtype=np.float32)

        # Apply flip first so that rotations happen around the flipped image
        if aug_params["flip"]:
            img = flip_img(img)
            center[0] = img_w - center[0] - 1
            gt_j2d = flip_kp(gt_j2d)
            gt_pose = flip_pose(gt_pose.copy())

        if aug_params["extreme_crop"]:
            scale *= 1.35

        scale *= aug_params["scale_noise"]
        center = center + aug_params["trans"] * scale * 200
        rot = aug_params["rot"]

        img_crop = crop(img, center, scale, [self.crop_size, self.crop_size], rot=rot).astype(np.float32)
        img_tensor = self.normalize(img_crop)

        # Normalize 2D joints to crop space and to [-0.5, 0.5]
        if rot != 0:
            crop_joints = transform_pts(gt_j2d.copy(), center, scale, [self.crop_size, self.crop_size], rot=rot, asint=False)
        else:
            crop_joints = crop_j2d(gt_j2d.copy(), center, scale, [self.crop_size, self.crop_size])
        crop_joints = crop_joints / self.crop_size - 0.5

        if gt_j3d is None and self.mano_layer is not None:
            mano_rot = rot_aa(gt_pose.copy(), rot)
            rotmat = torch.from_numpy(mano_rot.reshape(-1, 3)).float()
            betas_tensor = torch.from_numpy(betas.reshape(1, -1)).float()
            mano_out = self.mano_layer(
                global_orient=rotmat[:1].reshape(1, 1, 3),
                hand_pose=rotmat[1:].reshape(1, -1, 3),
                betas=betas_tensor,
                pose2rot=True,
            )
            gt_j3d = mano_out.joints.detach().cpu().numpy()[0]
        elif gt_j3d is None:
            raise KeyError("gt_j3d_wo_trans missing and MANO model not available to regenerate it.")

        img_center = sample.get("img_center")
        img_focal = sample.get("img_focal")
        if img_center is None or img_focal is None:
            img_center_tensor, img_focal_tensor = est_intrinsics(img.shape)
            img_center = img_center_tensor.numpy()
            img_focal = float(img_focal_tensor.item())

        frame = {
            "img": img_tensor,
            "center": torch.from_numpy(center).float(),
            "scale": torch.tensor(scale).float(),
            "img_focal": torch.tensor(img_focal).float(),
            "img_center": torch.tensor(img_center).float(),
            "gt_cam_j2d": torch.from_numpy(crop_joints).float(),
            "gt_cam_full_pose": torch.from_numpy(gt_pose).float(),
            "gt_cam_betas": torch.from_numpy(betas).float(),
            "gt_j3d_wo_trans": torch.from_numpy(gt_j3d).float(),
        }

        if aug_params["flip"]:
            frame["do_flip"] = torch.tensor(1.0)

        return frame


def build_dataloader(
    annotation_path: str,
    cfg,
    is_train: bool,
    image_root: Optional[str] = None,
) -> DataLoader:
    dataset = HaWoRDataset(annotation_path=annotation_path, cfg=cfg, is_train=is_train, image_root=image_root)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=is_train and cfg.TRAIN.SHUFFLE,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=cfg.GENERAL.PIN_MEMORY,
        drop_last=is_train,
        collate_fn=lambda batch: {"img": default_collate(batch)},
    )
    return dataloader

