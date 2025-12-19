import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================
# Streamlit config
# =================================================
st.set_page_config(
    page_title="Seed Species Classification",
    page_icon="🌳",
    layout="centered"
)

st.title("🌳 Seed Species Classification")
st.markdown(
    "Upload a seed image and select a trained model "
    "to **classify seed species**."
)

# =================================================
# Classes
# =================================================
CLASS_NAMES = [
    "Akasya (Acacia)",
    "Erguvan (Cercis siliquastrum)",
    "Gladiçya (Gleditsia triacanthos)",
    "Keçiboynuzu (Ceratonia siliqua)"
]
NUM_CLASSES = len(CLASS_NAMES)

# =================================================
# Image transform (training setup)
# =================================================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =================================================
# Model builders (FULL MODELS)
# =================================================
def build_mobilenet():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )
    return model


def build_resnet():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# =================================================
# MODEL LOADER (FINAL & SAFE)
# =================================================
@st.cache_resource
def load_model(model_name):

    if model_name == "MobileNetV3-Large":
        model = build_mobilenet()
        ckpt = torch.load("models/mobilenetv3_large_best.pt", map_location="cpu")
    else:
        model = build_resnet()
        ckpt = torch.load("models/resnet18_best.pt", map_location="cpu")

    # ---- state_dict ayıkla ----
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "state"]:
            if key in ckpt:
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # DataParallel temizliği
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ---- TAM MODELE YÜKLE ----
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    st.sidebar.success("✔ Model fully loaded")

    return model

# =================================================
# Sidebar
# =================================================
st.sidebar.header("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a trained model",
    ["MobileNetV3-Large", "ResNet18"]
)

model = load_model(model_name)

# =================================================
# Image upload
# =================================================
uploaded_file = st.file_uploader(
    "📤 Upload a seed image",
    type=["jpg", "jpeg", "png"]
)

# =================================================
# Prediction
# =================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))

    st.markdown("---")
    st.subheader("✅ Prediction Result")
    st.success(f"**Predicted Species:** {CLASS_NAMES[pred_idx]}")

    st.subheader("📊 Class Probabilities")

    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability (%)": probs * 100
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["Class"], df["Probability (%)"])
    ax.set_xlim(0, 100)

    for i, v in enumerate(df["Probability (%)"]):
        ax.text(v + 1, i, f"{v:.2f}%", va="center")

    st.pyplot(fig)