# 🧠 Tuberculosis Detection using Deep Learning

An end-to-end **Medical AI system** for detecting Tuberculosis (TB) from chest X-ray images using deep learning, with explainability using Grad-CAM.

---

## 🚀 Project Highlights

- 🔬 Multi-model comparison (**ResNet50, VGG16, EfficientNetB0**)
- 📊 Advanced evaluation (**ROC-AUC, F1-score, Confusion Matrix**)
- ⚖️ Handled **class imbalance** using dynamic class weights
- 🔥 Explainability with **Grad-CAM**
- 🖥️ Interactive **Streamlit web app**
- 🐳 Fully containerized using **Docker**

---

## 📊 Final Model Performance

| Metric | Value |
|--------|------|
| Model | VGG16 (Transfer Learning) |
| Accuracy | **90%** |
| ROC-AUC | **0.985 🔥** |
| Recall (Normal) | **1.00** |
| Recall (TB) | **0.88** |

### 🧠 Key Insight
Accuracy alone was misleading due to class imbalance.  
Applying class weights significantly improved recall and ROC-AUC.

---

## 🏗️ Project Architecture

```

tuberculosis-detection-v2/
│
├── src/
│   ├── components/        # data, model, training, evaluation
│   │   ├── model_builder.py
│   │   ├── data_loader.py
│   │   ├── gradcam.py
│   ├── pipeline/          # train & prediction pipeline
│   │   ├── train_pipeline.py
│   │   ├── evaluate.py
│   │   ├── gradcam_pipeline.py
│   ├── config/            # config.yaml
│   └── utils/
│
├── notebooks/             # EDA
├── app/                   # Streamlit app
├── models/                # saved models
├── logs/                  # training logs
├── artifacts/             # outputs (plots, metrics)
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- OpenCV  
- Matplotlib  
- Streamlit  
- Docker  

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/tuberculosis-detection-v2.git
cd tuberculosis-detection-v2

## Create Virtual Environment
```bash
conda create -n tb_v2_env python=3.10 -y
conda activate tb_v2_env
```

## Install Initial Dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python pillow tqdm streamlit ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Train Model
```bash
python -m src.pipeline.train_pipeline
```

### Evaluate Model
```bash
python -m src.pipeline.evaluate
```

### Run Streamlit App
 ```bash
streamlit run app/streamlit_app.py
```

## 🔥 Key Learnings

- Class imbalance can break model reliability
- Accuracy ≠ good model (use ROC-AUC, F1-score)
- Explainability is critical in medical AI
- Model debugging is as important as training

## 💡 Future Improvements

- Use larger datasets (NIH / CheXpert)
- Deploy on AWS / GCP
- Add model ensemble
- Integrate real-time API

## 👨‍💻 Author

Shubham Pandey
GitHub: https://github.com/shubhampandey97

## ⭐ Badges

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)