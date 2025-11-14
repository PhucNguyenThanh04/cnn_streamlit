# # test_inference.py
# import os
# import argparse
# from typing import List, Optional, Tuple
# from PIL import Image
# import csv
# import glob
#
# import torch
# import torch.nn.functional as F
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader, Dataset
#
# # import your model & config (adjust import path if needed)
# from model import SimpleCNN
# from cfg import Config
#
# # --- Transform (same as training) ---
# def get_transform(image_size: int):
#     return transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
# # --- Helper: load checkpoint weights into model ---
# def load_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
#     ckpt = torch.load(ckpt_path, map_location=device)
#     # checkpoint format used in your training: keys "model_state", "optimizer_state", "epoch", "best_acc"
#     state = ckpt.get("model_state", ckpt)  # fallback if someone saved plain state_dict
#     model.load_state_dict(state)
#     model.to(device)
#     model.eval()
#     return model
#
# # --- Predict single image ---
# def predict_one(model: torch.nn.Module, image_path: str, transform: transforms.Compose, device: torch.device) -> Tuple[int, List[float]]:
#     img = Image.open(image_path).convert("RGB")
#     tensor = transform(img).unsqueeze(0).to(device)  # batch size 1
#     with torch.no_grad():
#         logits = model(tensor)
#         probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
#         pred = int(logits.argmax(dim=1).cpu().item())
#     return pred, probs
#
# # --- Predict multiple images from a flat folder (no class subfolders) ---
# def predict_folder_flat(model: torch.nn.Module, folder: str, transform: transforms.Compose,
#                         device: torch.device, batch_size: int = 32, exts: List[str]=["jpg","jpeg","png"]) -> List[Tuple[str,int,List[float]]]:
#     # build list of image paths
#     paths = []
#     for ext in exts:
#         paths.extend(glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True))
#     paths = sorted(paths)
#     results = []
#     if not paths:
#         return results
#     # create a simple Dataset that applies transform
#     class SimpleImageDataset(Dataset):
#         def __init__(self, paths, transform):
#             self.paths = paths
#             self.transform = transform
#         def __len__(self): return len(self.paths)
#         def __getitem__(self, idx):
#             p = self.paths[idx]
#             img = Image.open(p).convert("RGB")
#             return self.transform(img), p
#
#     ds = SimpleImageDataset(paths, transform)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     with torch.no_grad():
#         for batch in loader:
#             imgs, paths_batch = batch
#             imgs = imgs.to(device)
#             logits = model(imgs)
#             probs_batch = F.softmax(logits, dim=1).cpu().numpy()
#             preds = logits.argmax(dim=1).cpu().numpy()
#             for p, pred, probs in zip(paths_batch, preds, probs_batch):
#                 results.append((str(p), int(pred), probs.tolist()))
#     return results
#
# # --- Predict folder organized as ImageFolder (subfolders = classes) ---
# def predict_folder_imagefolder(model: torch.nn.Module, folder: str, transform: transforms.Compose,
#                                device: torch.device, batch_size: int = 32) -> Tuple[List[Tuple[str,int,List[float]]], List[str]]:
#     ds = datasets.ImageFolder(root=folder, transform=transform)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     # class names from ImageFolder
#     class_names = [c for c, _ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]
#     results = []
#     with torch.no_grad():
#         for imgs, _ in loader:
#             imgs = imgs.to(device)
#             logits = model(imgs)
#             probs_batch = F.softmax(logits, dim=1).cpu().numpy()
#             preds = logits.argmax(dim=1).cpu().numpy()
#             # we cannot get file path from ImageFolder loader easily without custom dataset,
#             # but we can rebuild by iterating ds.samples if needed. For simplicity, return only preds/probs per item order.
#             for pred, probs in zip(preds, probs_batch):
#                 results.append((None, int(pred), probs.tolist()))
#     return results, class_names
#
# # --- Save results to CSV ---
# def save_results_csv(results: List[Tuple[str,int,List[float]]], out_csv: str, class_names: Optional[List[str]] = None):
#     header = ["image_path", "pred_index", "pred_label", "probs"]
#     with open(out_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         for path, pred, probs in results:
#             label = class_names[pred] if class_names and pred < len(class_names) else str(pred)
#             writer.writerow([path, pred, label, ";".join([f"{p:.6f}" for p in probs])])
#
# # --- CLI / main ---
# def main():
#     parser = argparse.ArgumentParser(description="Inference script — single image or folder")
#     parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint (last_model.pt / best_model.pt)")
#     parser.add_argument("--input", "-i", required=True, help="Path to image file or folder")
#     parser.add_argument("--image-size", type=int, default=224)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto-detect")
#     parser.add_argument("--out-csv", type=str, default=None, help="If set, save predictions to CSV")
#     parser.add_argument("--class-names", type=str, default=None,
#                         help="Optional comma-separated class names to map indices to labels (e.g. 'no_stone,stone')")
#     args = parser.parse_args()
#
#     device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
#     transform = get_transform(args.image_size)
#
#     # create model instance (match how you constructed during training)
#     # use cfg.num_classes if available, else default to 2
#     try:
#         cfg = Config()  # if your Config has defaults
#         num_classes = getattr(cfg, "num_classes", 2)
#     except Exception:
#         num_classes = 2
#
#     model = SimpleCNN(num_classes=num_classes, in_channels=3)
#     model = load_model_from_checkpoint(model, args.checkpoint, device)
#
#     # build class names if provided
#     class_names = None
#     if args.class_names:
#         class_names = [s.strip() for s in args.class_names.split(",")]
#
#     results = []
#
#     if os.path.isfile(args.input):
#         # single image
#         pred, probs = predict_one(model, args.input, transform, device)
#         label = class_names[pred] if class_names and pred < len(class_names) else str(pred)
#         print(f"{args.input} -> pred: {pred} ({label}), probs: {probs}")
#         results.append((args.input, pred, probs))
#
#     elif os.path.isdir(args.input):
#         # try ImageFolder structure detection: contains subfolders?
#         subdirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
#         if subdirs:
#             # If subdirs exist and look like class folders, use ImageFolder
#             print("Detected subdirectories — using ImageFolder mode.")
#             res_ifolder, detected_names = predict_folder_imagefolder(model, args.input, transform, device, batch_size=args.batch_size)
#             # res_ifolder items have path=None (keeps order); here we map indices -> names
#             # Build results with dummy paths (index)
#             for idx, item in enumerate(res_ifolder):
#                 _, pred, probs = item
#                 label = detected_names[pred] if detected_names and pred < len(detected_names) else str(pred)
#                 print(f"[{idx}] pred: {pred} ({label}), probs: {probs}")
#                 results.append((f"index_{idx}", pred, probs))
#             # if user supplied class-names override, prefer that for CSV label mapping
#             if class_names is None:
#                 class_names = detected_names
#         else:
#             # flat folder — process all images recursively
#             print("No subdirectories detected — processing all images in folder (flat mode).")
#             res = predict_folder_flat(model, args.input, transform, device, batch_size=args.batch_size)
#             for path, pred, probs in res:
#                 label = class_names[pred] if class_names and pred < len(class_names) else str(pred)
#                 print(f"{path} -> pred: {pred} ({label}), probs: {probs}")
#             results = res
#     else:
#         raise FileNotFoundError(f"Input path not found: {args.input}")
#
#     # optionally save CSV
#     if args.out_csv and results:
#         save_results_csv(results, args.out_csv, class_names)
#         print(f"Saved results to {args.out_csv}")
#
# if __name__ == "__main__":
#     main()


# test_inference_show.py
import os
import argparse
from typing import List, Optional, Tuple
from PIL import Image
import csv
import glob
import math

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# import your model & config (adjust import path if needed)
from model import SimpleCNN
from cfg import Config

# ---------- constants ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LABEL_MAP = {0: "Normal", 1: "stone"}

# --- Transform (same as training) ---
def get_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

# --- denormalize tensor for display (C,H,W) or batch (B,C,H,W) ---
def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    tensor: C,H,W or B,C,H,W in range normalized by MEAN/STD
    returns uint8 numpy array in H,W,C
    """
    if tensor.dim() == 3:
        t = tensor.clone()
        for c in range(3):
            t[c] = t[c] * STD[c] + MEAN[c]
        arr = (t.permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype("uint8")
        return arr
    elif tensor.dim() == 4:
        outs = []
        for i in range(tensor.size(0)):
            outs.append(denormalize_tensor(tensor[i]))
        return outs
    else:
        raise ValueError("Unsupported tensor dim for denormalize")

# --- Helper: load checkpoint weights into model ---
def load_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # support both formats
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# --- Predict single image (path) ---
def predict_one(model: torch.nn.Module, image_path: str, transform: transforms.Compose, device: torch.device) -> Tuple[int, List[float]]:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # batch size 1
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(logits.argmax(dim=1).cpu().item())
    return pred, probs, tensor.squeeze(0)  # return tensor for display if needed

# --- Predict folder flat (returns list of tuples (path,pred,probs,tensor)) ---
def predict_folder_flat(model: torch.nn.Module, folder: str, transform: transforms.Compose,
                        device: torch.device, batch_size: int = 32, exts: List[str]=["jpg","jpeg","png"]) -> List[Tuple[str,int,List[float],torch.Tensor]]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True))
    paths = sorted(paths)
    results = []
    if not paths:
        return results

    class SimpleImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            p = self.paths[idx]
            img = Image.open(p).convert("RGB")
            return self.transform(img), p

    ds = SimpleImageDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for imgs, paths_batch in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            for p, pred, probs, t in zip(paths_batch, preds, probs_batch, imgs):
                results.append((str(p), int(pred), probs.tolist(), t.cpu()))
    return results

# --- Predict folder ImageFolder (returns list with real file paths) ---
def predict_folder_imagefolder(model: torch.nn.Module, folder: str, transform: transforms.Compose,
                               device: torch.device, batch_size: int = 32) -> Tuple[List[Tuple[str,int,List[float],torch.Tensor]], List[str]]:
    ds = datasets.ImageFolder(root=folder, transform=transform)
    # ds.samples is list of (path, class_idx)
    paths = [s[0] for s in ds.samples]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    results = []
    idx = 0
    with torch.no_grad():
        for imgs, _ in loader:
            bs = imgs.size(0)
            imgs = imgs.to(device)
            logits = model(imgs)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            for i in range(bs):
                p = paths[idx]
                pred = int(preds[i])
                probs = probs_batch[i].tolist()
                t = imgs[i].cpu()
                results.append((p, pred, probs, t))
                idx += 1
    class_names = [c for c, _ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]
    return results, class_names

# --- Save results to CSV ---
def save_results_csv(results: List[Tuple[str,int,List[float],torch.Tensor]], out_csv: str, class_names: Optional[List[str]] = None):
    header = ["image_path", "pred_index", "pred_label", "probs"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for path, pred, probs, _ in results:
            label = class_names[pred] if class_names and pred < len(class_names) else LABEL_MAP.get(pred, str(pred))
            writer.writerow([path, pred, label, ";".join([f"{p:.6f}" for p in probs])])

# --- Display a batch of images (list of tensors) with titles ---
def show_image_batch(results_batch: List[Tuple[str,int,List[float],torch.Tensor]], cols: int = 4, figsize: Tuple[int,int] = (12,8)):
    n = len(results_batch)
    rows = math.ceil(n / cols)
    plt.figure(figsize=figsize)
    for i, (path, pred, probs, tensor) in enumerate(results_batch):
        img = denormalize_tensor(tensor)  # HWC uint8
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        label = LABEL_MAP.get(pred, str(pred))
        plt.title(f"{label} ({pred})\n{probs[0]:.2f}/{probs[1]:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Inference script — display images and save results")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint (last_model.pt / best_model.pt)")
    parser.add_argument("--input", "-i", required=True, help="Path to image file or folder")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto-detect")
    parser.add_argument("--out-csv", type=str, default=None, help="If set, save predictions to CSV")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = get_transform(args.image_size)

    # create model instance (match training)
    try:
        cfg = Config()
        num_classes = getattr(cfg, "num_classes", 2)
    except Exception:
        num_classes = 2

    model = SimpleCNN(num_classes=num_classes, in_channels=3)
    model = load_model_from_checkpoint(model, args.checkpoint, device)

    results_all = []
    class_names = None

    if os.path.isfile(args.input):
        pred, probs, tensor = predict_one(model, args.input, transform, device)
        print(f"{args.input} -> pred: {pred} ({LABEL_MAP.get(pred)}), probs: {probs}")
        results_all.append((args.input, pred, probs, tensor))
        show_image_batch(results_all, cols=1, figsize=(4,4))

    elif os.path.isdir(args.input):
        subdirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        if subdirs:
            print("Detected subdirectories — using ImageFolder mode.")
            res_ifolder, class_names = predict_folder_imagefolder(model, args.input, transform, device, batch_size=args.batch_size)
            # display in batches
            for i in range(0, len(res_ifolder), args.batch_size):
                batch = res_ifolder[i:i+args.batch_size]
                for path, pred, probs, _ in batch:
                    print(f"{path} -> pred: {pred} ({LABEL_MAP.get(pred)}), probs: {probs}")
                show_image_batch(batch, cols=min(4, args.batch_size), figsize=(12, math.ceil(len(batch)/4)*3))
            results_all = res_ifolder
        else:
            print("No subdirectories detected — processing all images in folder (flat mode).")
            res = predict_folder_flat(model, args.input, transform, device, batch_size=args.batch_size)
            for i in range(0, len(res), args.batch_size):
                batch = res[i:i+args.batch_size]
                for path, pred, probs, _ in batch:
                    print(f"{path} -> pred: {pred} ({LABEL_MAP.get(pred)}), probs: {probs}")
                show_image_batch(batch, cols=min(4, args.batch_size), figsize=(12, math.ceil(len(batch)/4)*3))
            results_all = res
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")

    # optionally save CSV
    if args.out_csv and results_all:
        save_results_csv(results_all, args.out_csv, class_names)
        print(f"Saved results to {args.out_csv}")

if __name__ == "__main__":
    main()
