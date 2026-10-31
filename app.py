import os

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

from seedify_core import SPECIES_ORDER, SeedifyPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
EXAMPLES = os.path.join(HERE, "examples")
WEIGHTS = os.path.join(MODELS, "seedify_shufflenetv2_x1_0_scaleaug.pt")
DINOV2_LOCAL = os.path.join(MODELS, "vit_small_patch14_dinov2.lvd142m.safetensors")  # optional; downloaded if absent

# =================================================
# Streamlit config
# =================================================
st.set_page_config(page_title="Seedify – Seed Species Classification", page_icon="🌳", layout="centered")
st.title("🌳 Seedify – Seed Species Classification")

st.markdown("""
This application identifies single seeds of four Fabaceae species:
- *Cercis siliquastrum*
- *Ceratonia siliqua*
- *Gleditsia triacanthos*
- *Robinia pseudoacacia*

**How it works.** The background of the uploaded photograph is removed with an ISNet matting model, the seed is
placed in a fixed-scale white circle (so that its true size is preserved), and a ShuffleNetV2 1.0× network trained
with scale augmentation predicts the species. Photographs that do not show a single seed, and seeds that do not
resemble the four species, are reported instead of being forced into one of the classes.

**For reliable results**, upload a photograph of **one seed** on a **plain, light background**, taken from about
**8 cm** above the seed, similar to the example images below.

**Academic Note:** This application is the product of a scientific study conducted by Safa Balekoğlu, Fatma Çalışkan,
Servet Çalışkan, Beyaz Başak Eskişehirli, Elif Kartal, and Zeki Özen. The authors are listed in alphabetical order by
surname and then by first name. The study is currently under review in a scientific journal.
""")


# =================================================
# Model (loaded once per server process)
# =================================================
@st.cache_resource(show_spinner="Loading models (first start may take a minute)...")
def load_predictor():
    return SeedifyPredictor(WEIGHTS, ood_dir=MODELS, dinov2_file=DINOV2_LOCAL, threads=2)


predictor = load_predictor()
if predictor.ood is None:
    st.info("The 'unrecognised seed' check is temporarily unavailable (its feature model could not be loaded). "
            "Predictions are still made, but seeds of other species will be assigned to one of the four classes.")

# =================================================
# Example images
# =================================================
EXAMPLE_FILES = {
    "Cercis siliquastrum": "cercis_siliquastrum.jpg",
    "Ceratonia siliqua": "ceratonia_siliqua.jpg",
    "Gleditsia triacanthos": "gleditsia_triacanthos.jpg",
    "Robinia pseudoacacia": "robinia_pseudoacacia.jpg",
}


@st.cache_data
def thumbnail(path, width=400):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((width, width * 3))
    return im


if "example" not in st.session_state:
    st.session_state.example = None

st.subheader("📸 Example Images")
cols = st.columns(4)
for col, (name, fn) in zip(cols, EXAMPLE_FILES.items()):
    with col:
        st.image(thumbnail(os.path.join(EXAMPLES, fn)), caption=f"*{name}*", width="stretch")
        if st.button("Predict", key=f"ex_{fn}", width="stretch"):
            st.session_state.example = name

# =================================================
# Image upload
# =================================================
st.markdown("Upload a seed image to **classify the seed species**.")
uploaded_file = st.file_uploader("📤 Upload a seed image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.session_state.example = None

source, caption = None, None
if uploaded_file is not None:
    source, caption = uploaded_file, "Uploaded image"
elif st.session_state.example is not None:
    name = st.session_state.example
    source, caption = os.path.join(EXAMPLES, EXAMPLE_FILES[name]), f"Example image: *{name}*"

# =================================================
# Prediction
# =================================================
if source is not None:
    with st.spinner("Removing background and classifying..."):
        res = predictor.analyze(source)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.image(thumbnail(source) if isinstance(source, str) else ImageOps.exif_transpose(Image.open(source)),
                 caption=caption, width="stretch")
    with c2:
        if res["circle"] is not None:
            st.image(res["circle"], caption="Background-free seed (model input)", width="stretch")

    if st.session_state.example is not None and st.button("Close example"):
        st.session_state.example = None
        st.rerun()

    st.subheader("Prediction Result")
    if res["status"] == "invalid":
        st.error(res["message"])
    elif res["status"] == "unknown":
        st.warning(res["message"])
        st.caption("No species is reported for this image. If it is one of the four species, retake the photograph "
                   "following the instructions above.")
    else:
        st.success(f"**Predicted species:** *{res['label']}*  (probability {res['probs'][res['label']] * 100:.1f}%)")

        st.subheader("📊 Class Probabilities")
        names = SPECIES_ORDER[::-1]
        vals = [res["probs"][n] * 100 for n in names]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(names, vals, color=["#3B6FB6" if n == res["label"] else "#B0B0B0" for n in names])
        ax.set_xlim(0, 105)
        ax.set_xlabel("Probability (%)")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontstyle="italic")
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.caption("Validated scope: single, non-touching seeds of the four species on a plain light background, photographed "
           "at a camera distance similar to the training images. Other conditions are outside the validated scope.")
