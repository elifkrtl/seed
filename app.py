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

st.markdown("""
In this application, the MobileNetV3-Large model was fine-tuned and evaluated using seed images belonging to the following four species:
- Cercis siliquastrum
- Ceratonia siliqua
- Gleditsia triacanthos
- Robinia pseudoacacia

Therefore, users of this application should upload an image of a seed that belongs to one of these four species only.
Please ensure that the uploaded image contains a single seed and that the seed is clearly visible in the image.

Below, example images are provided to illustrate the expected input format.

**Academic Note:** This application is the product of a scientific study conducted by Safa Balekoğlu, Fatma Çalışkan, Servet Çalışkan, Beyaz Başak Eskişehirli, Elif Kartal, and Zeki Özen. The authors are listed in alphabetical order by surname and then by first name. The study is currently under review in a scientific journal.
""")

# Example images
st.subheader("📸 Example Images")
col1, col2 = st.columns(2)
with col1:
    st.image("cercis_siliquastrum.jpg", caption="Cercis siliquastrum", width=200)
    if st.button("Predict Cercis siliquastrum"):
        st.session_state.example_image_path = "cercis_siliquastrum.jpg"
        st.session_state.example_class = "Cercis siliquastrum"
    
    st.image("gleditsia_triacanthos.jpg", caption="Gleditsia triacanthos", width=200)
    if st.button("Predict Gleditsia triacanthos"):
        st.session_state.example_image_path = "gleditsia_triacanthos.jpg"
        st.session_state.example_class = "Gleditsia triacanthos"

with col2:
    st.image("ceratonia_siliqua.jpg", caption="Ceratonia siliqua", width=200)
    if st.button("Predict Ceratonia siliqua"):
        st.session_state.example_image_path = "ceratonia_siliqua.jpg"
        st.session_state.example_class = "Ceratonia siliqua"
    
    st.image("robin_pseudoacacia.jpg", caption="Robinia pseudoacacia", width=200)
    if st.button("Predict Robinia pseudoacacia"):
        st.session_state.example_image_path = "robin_pseudoacacia.jpg"
        st.session_state.example_class = "Robinia pseudoacacia"

# =================================================
# Classes
# =================================================
CLASS_NAMES = [
    "Robinia pseudoacacia",
    "Cercis siliquastrum",
    "Gleditsia triacanthos",
    "Ceratonia siliqua"
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

# =================================================
# MODEL LOADER (FINAL & SAFE)
# =================================================
@st.cache_resource
def load_model():

    model = build_mobilenet()
    ckpt = torch.load("models/mobilenetv3_large_best.pt", map_location="cpu")

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

    return model

# =================================================
# Load Model
# =================================================
model = load_model()

# =================================================
# Image upload
# =================================================
st.markdown("Upload a seed image to **classify seed species**.")
uploaded_file = st.file_uploader(
    "📤 Upload a seed image",
    type=["jpg", "jpeg", "png"]
)

# Initialize session state for examples
if 'example_image_path' not in st.session_state:
    st.session_state.example_image_path = None
    st.session_state.example_class = None

# =================================================
# Prediction
# =================================================
if uploaded_file is not None or st.session_state.example_image_path is not None:

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")
    else:
        image = Image.open(st.session_state.example_image_path).convert("RGB")
        st.image(image, caption=f"Example Image: {st.session_state.example_class}", width="stretch")
        if st.button("Close Example"):
            st.session_state.example_image_path = None
            st.session_state.example_class = None
            st.rerun()

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