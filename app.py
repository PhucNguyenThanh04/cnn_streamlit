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


st.set_page_config(page_title="Ultrasound Inference", layout="centered")

st.title("Ultrasound: Predict 0 = Normal, 1 = stone")
st.markdown(
    "Upload an image (RGB). Model uses same preprocessing as training: resize -> toTensor -> ImageNet normalize."
)

# Sidebar - checkpoint & device
st.sidebar.header("Settings")
default_ckpt = "trained_models/last_model.pt"
checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)
device_option = st.sidebar.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
image_size = st.sidebar.number_input("Image size (px)", value=224, min_value=32, max_value=2048, step=1)

# Device selection logic
if device_option == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif device_option == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

st.sidebar.markdown(f"**Using device:** `{device}`")

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
    st.write(f"**Index:** {pred}")
    st.write(f"**Label:** {label}")
    st.write(f"**Probabilities:** {prob_str}")

    # show a small horizontal bar for probabilities
    st.subheader("Confidence")
    col1, col2 = st.columns(2)
    col1.metric("Normal (0)", f"{probs[0]:.4f}")
    col2.metric("stone (1)", f"{probs[1]:.4f}")

    # Option to save result CSV locally (download)
    if st.button("Download prediction as CSV"):
        import pandas as pd
        df = pd.DataFrame([{
            "image_name": getattr(uploaded_file, "name", "uploaded"),
            "pred_index": pred,
            "pred_label": label,
            "prob_0": probs[0],
            "prob_1": probs[1]
        }])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Click to download CSV", csv_bytes, file_name="prediction.csv", mime="text/csv")

else:
    if model is None:
        st.warning("Model not loaded — please check checkpoint path in the sidebar.")
    else:
        st.info("Upload an image to see prediction.")

st.markdown("---")
st.markdown("Notes: \n- Ensure checkpoint path is correct and compatible with training format (contains `model_state`).\n- If running on a headless server without GPU, set device to `cpu` in sidebar.")
