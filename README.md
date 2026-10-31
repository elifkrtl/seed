# 🌳 Seedify – Seed Species Classification (Streamlit App)

Live app: **https://seedify.streamlit.app**

Seedify identifies single seeds of four Fabaceae species from a smartphone photograph:

- *Cercis siliquastrum*
- *Ceratonia siliqua*
- *Gleditsia triacanthos*
- *Robinia pseudoacacia*

## How it works

The app runs exactly the pipeline evaluated in the accompanying study:

1. **EXIF orientation** of the uploaded photograph.
2. **Coarse seed localisation** at 0.25× scale (illumination-flattened dark-blob detection).
3. **ISNet matting** (`rembg`, `isnet-general-use`) on a 900 × 900 px full-resolution crop; mask clean-up (5 × 5 opening, largest component, hole filling).
4. **Fixed-scale white circle**: the seed is centred in a circle of radius 340 px (native resolution), resampled to 512 × 512 px; the surrounding paper is used only as a white reference (white balance to 235) and is removed from the image. True seed size is preserved.
5. **Classification** with **ShuffleNetV2 1.0×** (ImageNet-pretrained, fine-tuned on 1200 background-free seed images, random scale augmentation 0.6–1.1).

Input checks report instead of forcing a prediction when
- no seed is found,
- more than one seed is present,
- the seed is cut by the frame (camera too close),
- the seed does not resemble the four species (**unrecognised seed**; frozen DINOv2 ViT-S/14 features, cosine distance to the 5th nearest of 3600 reference embeddings; threshold = 95% acceptance of known seeds; no retraining).

## Performance (from the study)

| Setting | Accuracy |
|---|---|
| Five-fold cross-validation, same phone (Galaxy Note 10+), deployed architecture with scale augmentation | 98.67 ± 1.30% |
| External test, other phone (Galaxy S21 FE 5G, 177 images, different day) | 96.0% (95% CI 92.1–98.1%) |
| End-to-end time per photograph (CPU) | ≈ 1.1 s (ISNet ≈ 1.0 s) |

Unrecognised-seed check (proxy validation: each species in turn treated as unseen): about 71% of unseen-species seeds rejected while 95% of known seeds are accepted; on the other phone 86% of genuine seeds are accepted. It has not yet been validated on real non-target Fabaceae seeds.

**Validated scope:** one non-touching seed of the four species on a plain light background, photographed from about 8 cm (similar to the example images). Other conditions are outside the validated scope.

## Repository layout

```
app.py                     Streamlit user interface
seedify_core.py            Inference pipeline (segmentation, circle, classifier, checks)
models/
  seedify_shufflenetv2_x1_0_scaleaug.pt   Deployed classifier (5.2 MB)
  ood_bank.npz                            Reference embeddings for the unrecognised-seed check
  ood_threshold.json                      Thresholds and validation summary
examples/                  Four example photographs (one per species)
requirements.txt
.streamlit/config.toml
```

Downloaded automatically at first start (not stored in the repository):
- ISNet weights `isnet-general-use.onnx` (≈ 179 MB) by `rembg` into `~/.u2net`
- DINOv2 weights `vit_small_patch14_dinov2.lvd142m` (≈ 88 MB) from the Hugging Face Hub via `timm`

If the DINOv2 weights cannot be downloaded, the app still classifies seeds but shows a notice that the unrecognised-seed check is unavailable. To run fully offline, place `vit_small_patch14_dinov2.lvd142m.safetensors` in `models/`.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Programmatic use:

```python
from seedify_core import SeedifyPredictor
p = SeedifyPredictor("models/seedify_shufflenetv2_x1_0_scaleaug.pt", ood_dir="models")
r = p.analyze("photo.jpg")   # r["status"] in {"ok", "unknown", "invalid"}, r["label"], r["probs"], r["circle"]
```

## 📜 Academic Note

This application is the product of a scientific study conducted by Safa Balekoğlu, Fatma Çalışkan, Servet Çalışkan, Beyaz Başak Eskişehirli, Elif Kartal, and Zeki Özen. The authors are listed in alphabetical order by surname and then by first name. The study is currently under review in a scientific journal.

## 📜 License

This project is intended for **academic and educational use**.
