# app.py
import io
import os
from typing import Tuple, List

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# adjust import paths if necessary
from model import SimpleCNN
from cfg import Config

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LABEL_MAP = {0: "Normal", 1: "stone"}

def get_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 2) -> torch.nn.Module:
    model = SimpleCNN(num_classes=num_classes, in_channels=3)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(model: torch.nn.Module, pil_image: Image.Image, transform: transforms.Compose, device: torch.device) -> Tuple[int, List[float]]:
    img_t = transform(pil_image).unsqueeze(0).to(device)  # 1, C, H, W
    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(logits.argmax(dim=1).cpu().item())
    return pred, probs


path_imgapp = "img_app"
st.set_page_config(page_title="Ultrasound Inference", layout="centered")

st.title("Data Mining")

st.header("I. Xây dựng và huấn luyện mô hình dự đoán ảnh siêu âm sỏi thận")
st.write(
    """
       #### 1. Chuẩn bị dữ liệu:
       - tải dữ liệu từ: https://data.mendeley.com/datasets/h6jc4xm4py/1
       - chia dữ liệu ra 3 tập: train, val, test. 
    """
)
st.image(os.path.join(path_imgapp, "splip.png"))
st.write(
    """
        #### 2. Tạo lớp dataloader để load ảnh và nhãn để huấn luyện mô hinh.
    """
)
st.code(
    """
    class UltrasoundDataLoader(Dataset):
        def __init__(
            self,
            root: str = "./Ultrasound_Stone_No_Stone",
            mode: str = "train",
            transform=None,
            cache_list: bool = True,
            return_path: bool = False
        ):
            self.categories = ["Normal", "stone"]
            self.transform = transform
            self.return_path = return_path
            self.image_paths: List[str] = []
            self.labels: List[int] = []
    
            data_path_mode = os.path.join(root, mode)
            assert os.path.isdir(data_path_mode), f"Invalid dataset split path: {data_path_mode}"
    
            cache_file = os.path.join(root, f"_{mode}_cache.pkl")
            if cache_list and os.path.exists(cache_file):
                import pickle
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                self.image_paths, self.labels = data["paths"], data["labels"]
            else:
                for label, category in enumerate(self.categories):
                    category_path = os.path.join(data_path_mode, category)
                    if not os.path.exists(category_path):
                        logging.warning(f"Missing category folder: {category_path}")
                        continue
                    for file in os.listdir(category_path):
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.image_paths.append(os.path.join(category_path, file))
                            self.labels.append(label)
    
                if cache_list:
                    import pickle
                    with open(cache_file, "wb") as f:
                        pickle.dump({"paths": self.image_paths, "labels": self.labels}, f)
    
            # Shuffle once
            data = list(zip(self.image_paths, self.labels))
            random.shuffle(data)
            self.image_paths, self.labels = zip(*data)
            self.image_paths, self.labels = list(self.image_paths), list(self.labels)
    
            logging.info(f"[{mode}] Found {len(self.image_paths)} samples ({len(self.categories)} classes)")
    
        def __len__(self) -> int:
            return len(self.labels)
    
        def __getitem__(self, index: int) -> Tuple:
            path = self.image_paths[index]
            label = self.labels[index]
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to load image {path}: {e}")
                # # fallback: create dummy black image to keep batch consistent
                # import numpy as np
                # image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                pass
            if self.transform:
                image = self.transform(image)
    
            return (image, label, path) if self.return_path else (image, label)
    """, language="python"
)
st.write("#### 3. xây dựng mạng cnn")
st.code(
    """
    class SimpleCNN(nn.Module):
        def __init__(
            self,
            num_classes: int = 2,
            in_channels: int = 3,
            channels: Optional[List[int]] = None,
            dropout: float = 0.5,
        ) -> None:
            super().__init__()
            if channels is None:
                channels = [32, 64, 128]
    
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.channels = channels
            self.dropout = dropout
    
            # Feature extractor
            blocks = []
            in_c = in_channels
            for out_c in channels:
                blocks.append(self._make_block(in_c, out_c))
                in_c = out_c
            self.features = nn.Sequential(*blocks)
    
            # Pool & mlp
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels[-1], max(32, channels[-1] // 2)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(max(32, channels[-1] // 2), num_classes),
            )
    
            # Initialize weights
            self.reset_parameters()
    
        def _make_block(self, in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)  # shape (B, channels[-1])
            x = self.classifier(x)
            return x
    """, language="python"
)
st.image(os.path.join(path_imgapp, "model.png"))

st.write(" #### 4. huấn luyện mô hình")
st.write("Hàm loss và optimizer")
st.code(
    """
        self.criterion = nn.CrossEntropyLoss()
        param_groups = self._get_param_groups(weight_decay=getattr(cfg, "weight_decay", 1e-4))

        self.optimizer = optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    """
)
st.write("Đồ thị biểu diễn hàm Loss khi huấn luyện mô hình")
st.image(os.path.join(path_imgapp, "trainloader.png"), caption="Đồ thị biểu diễn hàm Loss khi huấn luyện mô hình")
st.write("Đồ thị biểu diễn độ chính xác của mô hình trong quá trình huấn luyện")
st.image(os.path.join(path_imgapp, "testloaderr.png"), caption="Đồ thị biểu diễn độ chính xác của mô hình trong quá trình huấn luyện")
st.write("Sau khi quá trình huấn luyện kết thúc, các trọng số của mô hình sẽ được lưu vào best_model.pt và last_model.pt")
st.image(os.path.join(path_imgapp, "fil.png"), caption="Mô hình")

st.header("II. Chạy mô hình trên streamlit")

st.markdown(
    "Upload an image (RGB)"
)

default_ckpt = "trained_models/last_model.pt"
checkpoint_path = default_ckpt
device = torch.device("cpu")
image_size = 224


# Load model (cached)
try:
    # try to get num_classes from Config if available
    try:
        cfg = Config()
        num_classes = getattr(cfg, "num_classes", 2)
    except Exception:
        num_classes = 2
    model = load_model(checkpoint_path, device, num_classes=num_classes)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    model = None

# File uploader
uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Read image bytes -> PIL
    img_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Display original image
    st.subheader("Input image")
    st.image(pil_img, use_column_width=True)

    # Prepare transform & predict
    transform = get_transform(image_size)

    try:
        pred, probs = predict_image(model, pil_img, transform, device)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    label = LABEL_MAP.get(pred, str(pred))
    prob_str = " / ".join([f"{p:.4f}" for p in probs])

    st.subheader("Prediction")
    st.subheader(f"Label: {label}")
    # st.write(f"**Probabilities:** {prob_str}")

    # show a small horizontal bar for probabilities
    st.subheader("Confidence")
    col1, col2 = st.columns(2)
    col1.metric("Normal (0)", f"{probs[0]:.4f}")
    col2.metric("stone (1)", f"{probs[1]:.4f}")

    # Option to save result CSV locally (download)
    # if st.button("Download prediction as CSV"):
    #     import pandas as pd
    #     df = pd.DataFrame([{
    #         "image_name": getattr(uploaded_file, "name", "uploaded"),
    #         "pred_index": pred,
    #         "pred_label": label,
    #         "prob_0": probs[0],
    #         "prob_1": probs[1]
    #     }])
    #     csv_bytes = df.to_csv(index=False).encode("utf-8")
    #     st.download_button("Click to download CSV", csv_bytes, file_name="prediction.csv", mime="text/csv")

else:
    if model is None:
        st.warning("Model not loaded — please check checkpoint path in the sidebar.")
    else:
        st.info("Upload an image to see prediction.")
