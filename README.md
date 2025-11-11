# Multimodal Skin Lesion Classification

> Early melanoma detection through multimodal learning over dermatoscopic imagery and patient metadata.

This repository implements an end-to-end pipeline for the **HAM10000** dataset, unifying dermatoscopic image embeddings and structured metadata to perform both **7-class lesion diagnosis** and **binary malignant vs. benign screening**. The project covers preprocessing, feature extraction, classical and neural modeling, evaluation, interpretability, and reporting, all reproducibly orchestrated via CLI scripts.

---

## 📁 Project Structure

```
├── DATASET/                 # HAM10000 images & metadata CSV
├── artifacts/               # Generated plots, metrics, logs, cached embeddings
│   ├── eda/                 # Exploratory data analysis visuals
│   ├── metrics/             # CSV summaries, bar/radar charts, significance tests
│   ├── roc_curves/          # ROC curves for each model
│   ├── pr_curves/           # Precision-recall curves
│   ├── gradcam/             # Grad-CAM heatmaps
│   ├── shap/                # SHAP plots & feature importances
│   └── logs/                # env_info.json, pip_freeze.txt, preprocessing logs
├── configs/
│   ├── project_config.json  # Global config (e.g., seeds)
│   └── seed_config.json     # Deterministic random seed definitions
├── data/                    # Train/val/test CSV splits (multiclass & binary)
├── models/                  # Saved models (.joblib, .pt) and cached features (.npz)
├── notebooks/
│   └── 01_end_to_end.ipynb  # Guided notebook walkthrough
├── scripts/                 # CLI tooling for preprocessing, training, evaluation
├── utils/                   # Shared helpers (metrics, SHAP, Grad-CAM, late fusion, seeds)
├── .venv/                   # Local virtual environment (not tracked in VCS)
├── requirements.txt         # Pinned Python dependencies
├── setup_venv.sh            # Convenience script to create & populate .venv
└── README.md
```

---

## 🛠️ Environment Setup

```bash
# (Optional) create .venv using the helper script
chmod +x setup_venv.sh
./setup_venv.sh          # runs: python3 -m venv .venv && pip install -r requirements.txt

# activate the environment
source .venv/bin/activate

# or manually install if preferred
pip install -r requirements.txt
```

**Pinned stack:** Python ≥ 3.11 (tested), PyTorch 2.4.0, torchvision 0.19.0, scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4, shap 0.45.0, seaborn 0.13.2, matplotlib 3.9.2, OpenCV 4.10, tqdm 4.66.5.

---

## 📦 Dataset Placement

1. Download HAM10000 from the official source (ISIC archive / Kaggle). The Kaggle mirror is available here: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).  
2. Place the files as follows:

```
DATASET/
│── HAM10000_metadata.csv
│── HAM10000_images_part_1/
│   └── *.jpg
└── HAM10000_images_part_2/
    └── *.jpg
```

3. `scripts/preprocess_data.py` resolves absolute paths to these directories. If you use a different layout, edit the `DATASET_ROOT` constants within that script.

---

## 🚀 Run the Pipeline

> Activate the virtual environment before executing any CLI script.

```bash
source .venv/bin/activate

# 1. Generate splits, binary labels, diagnostics
python scripts/preprocess_data.py

# 2. Train image-only classical models on ResNet50 embeddings
python scripts/train_image_only.py

# 3. Train metadata-only classical models
python scripts/train_metadata_only.py

# 4. Train fusion models (classical + Fusion MLP)
python scripts/train_fusion.py

# 5. (Optional) regenerate CNN embeddings (e.g., switch backbone)
python scripts/extract_embeddings.py --model resnet50 --batch_size 32

# 6. Comprehensive evaluation, interpretability, subgroup analysis
python scripts/evaluate.py --debug

# Optional utilities
python scripts/run_weighted_late_fusion.py          # weighted ensemble experiments
python scripts/run_binary_significance.py          # McNemar & Friedman tests
python scripts/run_shap_extended.py                # extended SHAP waterfall plots
python scripts/run_image_pca_update.py             # PCA refresh + retraining
python scripts/aggregate_results.py                # leaderboard + visual summaries
```

---

## 📓 Notebook Companion

- **`notebooks/01_end_to_end.ipynb`** provides a narrative walkthrough covering preprocessing summaries, sample visualizations, and model performance snapshots. It’s ideal for newcomers wanting an interactive tour before diving into the CLI workflow.

---

## 🗂️ Outputs & Artifacts

- `artifacts/eda/` – Class distributions, age/localization plots, correlation heatmaps, PCA variance & CI charts.  
- `artifacts/evaluation/multiclass/` – Confusion matrices, ROC/PR curves, SHAP plots, subgroup analyses, Grad-CAM galleries, and metrics for the seven-class task.  
- `artifacts/evaluation/binary/` – Analogous artifacts for the malignant-vs-benign task, including weighted late-fusion outputs and statistical tests.  
- `artifacts/evaluation/summary/` – Aggregated leaderboards (`model_comparison.csv`, bar/radar charts) spanning both tasks.  
- `artifacts/eda/` – Exploratory data analysis visuals (class balance, correlation heatmaps, PCA variance, etc.).  
- `artifacts/cache/` – Cached image embeddings for train/val/test splits (ResNet50, VGG16 variants).  
- `artifacts/logs/` – `env_info.json`, `pip_freeze.txt`, preprocess/train/evaluate logs for reproducibility.

Example interpretability artifact:

![Grad-CAM example](artifacts/gradcam/grad_cam_tp_1.png)

---

## 🔁 Reproducibility

- **Seeds:** All scripts load deterministic seeds from `configs/seed_config.json` / `configs/project_config.json`.  
- **Environment Logs:** `artifacts/logs/env_info.json` captures Python, PyTorch, scikit-learn versions; `pip_freeze.txt` snapshots installed packages after setup.  
- **Configurable CLI:** Every script exposes arguments (`--data-dir`, `--models-dir`, etc.) to tailor experiments while maintaining a traceable command history.

---

## 📚 Citations & References

- **Dataset:** Tschandl, Rosendahl, and Kittler. “The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.” *Scientific Data*, 2018.  
- **CNN Backbone:** Kaiming He et al. “Deep Residual Learning for Image Recognition.” *CVPR*, 2016 (ResNet50).  
- **Interpretability:** Lundberg & Lee. “A Unified Approach to Interpreting Model Predictions.” *NeurIPS*, 2017 (SHAP).  
- **Grad-CAM:** Selvaraju et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” *ICCV*, 2017.

---


