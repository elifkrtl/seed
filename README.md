# 🌳 Seed Species Classification – Streamlit App

This Streamlit application performs **seed species classification** from images using deep learning models.

The app supports **two trained CNN architectures**:
- **MobileNetV3-Large**
- **ResNet18**

Users can upload an image, select a model, and obtain:
- The **predicted seed species**
- **Class probability distribution** visualized as a bar chart

---

## 🌲 Supported Seed Species (TR + Latin)

- **Akasya (Acacia)**
- **Erguvan (Cercis siliquastrum)**
- **Gladiçya (Gleditsia triacanthos)**
- **Keçiboynuzu (Ceratonia siliqua)**

---

## 📁 Project Structure

```text
seed/
│
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── mobilenetv3_large_best.pt
    └── resnet18_best.pt
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

- Both **MobileNetV3-Large** and **ResNet18** use **ImageNet-pretrained backbones**
- Only the **final classification layers** were fine-tuned on the seed dataset
- This design ensures stable inference while keeping checkpoints lightweight

**Academic statement you may use:**

> *Both MobileNetV3-Large and ResNet18 use ImageNet-pretrained backbones, while only the final classification layers were fine-tuned on the target dataset.*

---

## 📊 Output

- Predicted class shown clearly
- Class probabilities (%) displayed with a horizontal bar chart
- Suitable for:
  - Academic demos
  - Student projects
  - Streamlit Cloud deployment

---

## 🚀 Possible Extensions

- Grad-CAM visual explanations
- Top-3 predictions
- Confidence donut / gauge charts
- TR / Latin language toggle
- Streamlit Cloud deployment

---

## 📜 License

This project is intended for **academic and educational use**.
