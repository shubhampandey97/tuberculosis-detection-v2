# Tuberculosis Detection Using Deep Learning

```

tuberculosis-detection-v2/
│
├── src/
│   ├── components/        # data, model, training, evaluation
│   ├── pipeline/          # train & prediction pipeline
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

## Create Virtual Environment
```bash
conda create -n tb_v2_env python=3.10 -y
conda activate tb_v2_env
```

## Install Initial Dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python pillow tqdm streamlit ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121