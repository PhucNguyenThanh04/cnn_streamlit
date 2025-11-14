#conda idvision
from __future__ import annotations
import argparse
import logging
import os
import shutil
import csv
from typing import Optional, Sequence, List, Dict, Any, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_loader import UltrasoundDataLoader
from model import SimpleCNN
from cfg import Config
from set_logging import setup_logging


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DatasetFactory:
    @staticmethod
    def create(root: str, mode: str, transform: Optional[Callable] = None) -> Dataset:
        try:
            logging.info("Using UltrasoundDataloader")
            return UltrasoundDataLoader(root=root, mode=mode, transform=transform)
        except Exception:
            # fallback to ImageFolder assuming standard structure:
            # folder = os.path.join(root, "train" if mode == "train" else "val")
            # logging.info(f"UltrasoundDataLoader not found — falling back to ImageFolder at {folder}")
            # if transform is None:
            #     transform = transforms.Compose([
            #         transforms.Resize((224, 224)),
            #         transforms.ToTensor()
            #     ])
            # # assert folder exists
            # if not os.path.isdir(folder):
            #     raise FileNotFoundError(f"Expected folder {folder} for ImageFolder fallback.")
            # return datasets.ImageFolder(root=folder, transform=transform)
            logging.info("Using Image folder")
            exit(0)


class TensorBoardWriter:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_figure(self, tag: str, figure: plt.Figure, global_step: int) -> None:
        self.writer.add_figure(tag, figure, global_step)

    def close(self) -> None:
        self.writer.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str]) -> plt.Figure:
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize
    with np.errstate(all='ignore'):
        cm_norm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = np.nanmax(cm_norm) / 2. if cm_norm.size else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j] if cm_norm.size else 0.0
            color = "white" if val > threshold else "black"
            plt.text(j, i, f"{cm[i, j]}\n{val:.2f}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class Trainer:
    def __init__(self,
                 cfg: Config,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 writer: TensorBoardWriter
                 ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer

        self.criterion = nn.CrossEntropyLoss()
        param_groups = self._get_param_groups(weight_decay=getattr(cfg, "weight_decay", 1e-4))

        self.optimizer = optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )

        self.device_type = "cuda" if cfg.device.type == "cuda" else "cpu"

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.epochs), eta_min=1e-6)

        self.best_acc: float = 0.0
        self.start_epoch: int = 0
        self.num_iters = len(self.train_loader)
        os.makedirs(cfg.trained_models_dir, exist_ok=True)



    def _get_param_groups(self, weight_decay: float = 1e-4):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or ("bn" in name.lower()) or ("norm" in name.lower()):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decat": 0.0},
        ]

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_acc": self.best_acc
        }
        last_path = os.path.join(self.cfg.trained_models_dir, "last_model.pt")
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(self.cfg.trained_models_dir, "best_model.pt")
            torch.save(state, best_path)
        logging.info("Saved checkpoint")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model_state"])
        try:
            self.optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        except Exception as e:
            logging.warning(f"Failed load optimizer: {e}")
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_acc = ckpt.get("best_acc", 0.0)
        logging.info("loaded checkponit")

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        running_loss = 0.0
        n_samples = 0
        processbar_train = tqdm(self.train_loader,
                                desc=f"Train Epoch {epoch+1}/{self.cfg.epochs}",
                                leave=True, colour="green")
        for iter,  batch in enumerate(processbar_train):
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format from dataloader")
            images = images.to(self.cfg.device)
            labels = labels.to(self.cfg.device)

            #forward
            output = self.model(images)
            loss_value = self.criterion(output, labels)
            processbar_train.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}"\
                                             .format(epoch+1, self.cfg.epochs, iter+1, self.num_iters, loss_value))
            self.writer.add_scalar("Train/Loss", loss_value.item(), epoch * self.num_iters + iter)
            #backward
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss_value.item()) * batch_size
            n_samples += batch_size
        epoch_loss = running_loss / max(1, n_samples)
        logging.info(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}")

    def val_epoch(self, epoch: int) -> None:
        self.model.eval()
        all_pres: List[int] = []
        all_labels: List[int] = []
        running_loss = 0.0
        n_samples = 0

        processbar_val = tqdm(self.val_loader,
                                desc=f"Val Epoch {epoch+1}/{self.cfg.epochs}",
                                leave=True, colour="red")
        with torch.no_grad():
            for iter, batch in enumerate(processbar_val):
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    raise RuntimeError("Unexpected batch format from dataloader")
                images = images.to(self.cfg.device)
                labels = labels.to(self.cfg.device)

                pre = self.model(images)
                loss_val = self.criterion(pre, labels)
                probs = torch.softmax(pre, dim=1)
                indices = torch.argmax(probs, dim=1).cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                all_pres.extend(indices.tolist())
                all_labels.extend(labels_cpu.tolist())

        acc = accuracy_score(all_labels, all_pres)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, acc))
        self.writer.add_scalar("Val/Accuracy", acc, epoch)
        self.writer.add_scalar("Val/Loss", loss_val, epoch)

        if (epoch + 1) % self.cfg.save_every == 0:
            self.save_checkpoint(epoch + 1, is_best=False)
        is_best = acc > self.best_acc
        if is_best:
            self.best_acc = acc
            self.save_checkpoint(epoch, is_best=True)




    def fit(self) -> None:
        if self.cfg.checkpoint:
            self.load_checkpoint(self.cfg.checkpoint)


        for epoch in range(self.start_epoch, self.cfg.epochs):
            #train
            self.train_epoch(epoch)
            #val
            self.val_epoch(epoch)

        logging.info(f"Training finished. Best accuracy: {self.best_acc:.4f}")




def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train SimpleCNN (RGB, 2-class) — OOP style")
    parser.add_argument("--root", "-r", type=str, default="./Ultrasound_Stone_No_Stone", help="Root folder containing train/val")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--image-size", "-i", type=int, default=224)
    parser.add_argument("--logging-dir", "-l", type=str, default="runs")
    parser.add_argument("--trained-models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Disable AMP even if CUDA available")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        logging_dir=args.logging_dir,
        trained_models_dir=args.trained_models,
        checkpoint=args.checkpoint,
        lr=args.lr,
        momentum=0.9,
        num_workers=args.num_workers,
        seed=args.seed,
        num_classes=2,
        device=device,
        use_amp=not args.no_amp,
        save_every=1
    )
    return cfg


def main():
    setup_logging()
    cfg = parse_args()
    logging.info(f"Config: {cfg}")
    set_seed(cfg.seed)

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # datasets
    train_dataset = DatasetFactory.create(cfg.root, mode="train", transform=transform)
    val_dataset = DatasetFactory.create(cfg.root, mode="val", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if os.path.isdir(cfg.logging_dir):
        logging.info(f"Removing existing logging dir: {cfg.logging_dir}")
        shutil.rmtree(cfg.logging_dir)
    os.makedirs(cfg.logging_dir, exist_ok=True)
    os.makedirs(cfg.trained_models_dir, exist_ok=True)

    model = SimpleCNN(num_classes=cfg.num_classes, in_channels=3)
    model.to(cfg.device)
    logging.info("Model created:\n%s", model)

    writer = TensorBoardWriter(cfg.logging_dir)
    trainer = Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader, writer=writer)

    try:
        trainer.fit()
    finally:
        writer.close()

if __name__ == '__main__':
    main()
