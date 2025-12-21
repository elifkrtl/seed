# 🌳 Seed Species Classification – Streamlit App

This Streamlit application performs **seed species classification** from images using the MobileNetV3-Large deep learning model.

The app uses a **fine-tuned MobileNetV3-Large model** trained on seed images.

Users can upload an image and obtain:
- The **predicted seed species**
- **Class probability distribution** visualized as a bar chart

---

## 🌲 Supported Seed Species (Latin)

- **Cercis siliquastrum**
- **Ceratonia siliqua**
- **Gleditsia triacanthos**
- **Robinia pseudoacacia**

---

## 📁 Project Structure

```text
seed/
│
├── app.py
├── requirements.txt
├── README.md
├── cercis_siliquastrum.jpg
├── ceratonia_siliqua.jpg
├── gleditsia_triacanthos.jpg
├── robin_pseudoacacia.jpg
└── models/
    └── mobilenetv3_large_best.pt
```

---

## ⚙️ Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root directory:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🧠 Model Notes (Important)

<<<<<<< HEAD
- Both **MobileNetV3-Large** and **ResNet-18** use **ImageNet-pretrained backbones**
- Only the **final classification layers** were fine-tuned on the seed dataset
- This design ensures stable inference while keeping checkpoints lightweight

> *Both MobileNetV3-Large and ResNet18 use ImageNet-pretrained backbones, while only the final classification layers were fine-tuned on the target dataset.*
=======
- The **MobileNetV3-Large** uses an **ImageNet-pretrained backbone**
- Only the **final classification layer** was fine-tuned on the seed dataset
- This design ensures stable inference while keeping the checkpoint lightweight

**Academic statement you may use:**

> *The MobileNetV3-Large model uses an ImageNet-pretrained backbone, while only the final classification layer was fine-tuned on the target seed dataset.*
>>>>>>> 70614e7 (app_v2)

---

## 📊 Output

- Predicted class shown clearly
- Class probabilities (%) displayed with a horizontal bar chart
- Suitable for:
  - Academic demos
  - Student projects
  - Streamlit Cloud deployment

---

<<<<<<< HEAD
=======
## 🚀 Possible Extensions

- Grad-CAM visual explanations
- Top-3 predictions
- Confidence donut / gauge charts
- TR / Latin language toggle
- Streamlit Cloud deployment

---

## 📜 Academic Note

This application is the product of a scientific study conducted by Safa Balekoğlu, Fatma Çalışkan, Servet Çalışkan, Beyaz Başak Eskişehirli, Elif Kartal, and Zeki Özen. The authors are listed in alphabetical order by surname and then by first name. The study is currently under review in a scientific journal.

>>>>>>> 70614e7 (app_v2)
## 📜 License

This project is intended for **academic and educational use**.
