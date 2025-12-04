import argparse
import os
from typing import Dict, Tuple

import torch
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hawor.configs import get_config
from hawor.utils.pylogger import get_pylogger
from lib.datasets.hawor_dataset import build_dataloader
from lib.models.hawor import HAWOR

log = get_pylogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HaWoR with PyTorch")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--train-annotations", type=str, required=True, help="Path to training annotation file")
    parser.add_argument("--val-annotations", type=str, required=True, help="Path to validation annotation file")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory prepended to relative img_path entries")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for checkpoints and logs")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume training from checkpoint path")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override number of GPUs from config (0 for CPU)")
    parser.add_argument("--no-image-log", action="store_true", help="Disable TensorBoard image logging for speed")
    return parser.parse_args()


def prepare_output_dirs(cfg, output_dir: str) -> Tuple[str, str]:
    summary_dir = os.path.join(output_dir, cfg.GENERAL.SUMMARY_DIR)
    checkpoint_dir = os.path.join(output_dir, cfg.GENERAL.CHECKPOINT_DIR)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return summary_dir, checkpoint_dir


def build_optimizer_and_scheduler(model: HAWOR, cfg):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.get_parameters()),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    scheduler = None
    if cfg.TRAIN.LR_SCHEDULER.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.TRAIN.NUM_EPOCHS, eta_min=cfg.TRAIN.MIN_LR
        )
    elif cfg.TRAIN.LR_SCHEDULER.lower() == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.TRAIN.LR_DECAY_STEPS, gamma=cfg.TRAIN.LR_DECAY_GAMMA
        )
    return optimizer, scheduler


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def save_checkpoint(
    checkpoint_path: str,
    model: HAWOR,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val: float,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
        },
        checkpoint_path,
    )


def load_checkpoint(path: str, model: HAWOR, optimizer, scheduler, scaler):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    best_val = checkpoint.get("best_val", float("inf"))
    log.info("Resumed from %s (epoch %d, global step %d)", path, start_epoch, global_step)
    return start_epoch, global_step, best_val


def train_one_epoch(
    model: HAWOR,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    global_step: int,
    cfg,
    writer: SummaryWriter,
    log_images: bool,
) -> int:
    model.train()
    for batch_idx, joint_batch in enumerate(dataloader):
        batch = move_batch_to_device(joint_batch["img"], device)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=cfg.GENERAL.MIXED_PRECISION and device.type == "cuda"):
            output = model.forward_step(batch, train=True)
            loss = model.compute_loss(batch, output, train=True)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        scaler.scale(loss).backward()

        if cfg.TRAIN.GRAD_CLIP_VAL and cfg.TRAIN.GRAD_CLIP_VAL > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)

        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        if writer is not None:
            writer.add_scalar("train/loss", loss.detach().item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if global_step % cfg.GENERAL.LOG_STEPS == 0 and log_images:
                model.tensorboard_logging(batch, output, global_step, writer=writer, train=True, render_log=True)

    return global_step


def evaluate(model: HAWOR, dataloader: DataLoader, device: torch.device, cfg, writer: SummaryWriter, global_step: int) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for joint_batch in dataloader:
            batch = move_batch_to_device(joint_batch["img"], device)
            output = model.forward_step(batch, train=False)
            loss = model.compute_loss(batch, output, train=False)
            total_loss += loss.item()
            num_batches += 1

    mean_loss = total_loss / max(1, num_batches)
    if writer is not None:
        writer.add_scalar("val/loss", mean_loss, global_step)
    return mean_loss


def main():
    args = parse_args()
    cfg = get_config(args.config, merge=True, update_cachedir=True)
    cfg.defrost()
    if args.num_gpus is not None:
        cfg.GENERAL.NUM_GPUS = args.num_gpus
    if args.image_root:
        cfg.DATASETS.IMAGE_ROOT = args.image_root
    cfg.freeze()

    summary_dir, checkpoint_dir = prepare_output_dirs(cfg, args.output_dir)
    writer = SummaryWriter(summary_dir)

    train_loader = build_dataloader(args.train_annotations, cfg, is_train=True, image_root=args.image_root)
    val_loader = build_dataloader(args.val_annotations, cfg, is_train=False, image_root=args.image_root)

    model = HAWOR(cfg)
    device = torch.device(
        "cuda"
        if cfg.GENERAL.ALLOW_CUDA and torch.cuda.is_available() and cfg.GENERAL.NUM_GPUS != 0
        else "cpu"
    )
    model.to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    scaler = amp.GradScaler(enabled=cfg.GENERAL.MIXED_PRECISION and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume_from_checkpoint:
        start_epoch, global_step, best_val = load_checkpoint(
            args.resume_from_checkpoint, model, optimizer, scheduler, scaler
        )

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        log.info("Starting epoch %d/%d", epoch + 1, cfg.TRAIN.NUM_EPOCHS)
        global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            global_step,
            cfg,
            writer,
            log_images=not args.no_image_log,
        )

        val_loss = evaluate(model, val_loader, device, cfg, writer, global_step)
        log.info("Validation loss after epoch %d: %.6f", epoch + 1, val_loss)

        if scheduler is not None:
            scheduler.step()

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        checkpoint_name = "epoch_%03d.ckpt" % (epoch + 1)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val)
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best.ckpt")
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val)
            log.info("Saved new best checkpoint to %s", best_path)

    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val)


if __name__ == "__main__":
    main()
