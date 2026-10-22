# Can Deep Learning Reliably Distinguish Tumor Types from MRI Images?

**Team SuperFrog** — Ethan Paredez and Sam Shema
COSC 50523: Deep Learning, Fall 2026, Texas Christian University

This project classifies brain MRI slices into three tumor types (meningioma, glioma, pituitary tumor). It compares a convolutional neural network trained from scratch with a pretrained ResNet-18, and it measures how splitting the data by patient versus by image changes the reported performance.

> For educational and research purposes only. These models are not intended for clinical diagnosis.

## Contents

| File | Description |
|---|---|
| `Ethan_Paredez_&_Sam_Shema_Brain_Tumor_MRI_Classification.ipynb` | The full pipeline: data download, exploration, patient-level split, models, training, evaluation, and figures. Saved with outputs. |
| `requirements.txt` | Python packages used. |

The dataset and trained model weights are **not** included. The notebook downloads the dataset automatically, and models are retrained when the notebook runs.

## How to run

The notebook is designed for Google Colab with a GPU.

1. Open `Ethan_Paredez_&_Sam_Shema_Brain_Tumor_MRI_Classification.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. Select a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run everything: **Runtime → Run all**.

The full run takes about 30–40 minutes on a T4 GPU. No Google Drive access or manual download is needed.

To run locally instead, install the packages with `pip install -r requirements.txt` and run the notebook in Jupyter. A CUDA GPU is strongly recommended.

## Dataset

Figshare Brain Tumor Dataset (J. Cheng): 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients (708 meningioma, 1,426 glioma, 930 pituitary). License: CC BY 4.0.

- Source: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- DOI: 10.6084/m9.figshare.1512427.v8

The notebook downloads the four ZIP archives through the Figshare API (about 880 MB) and extracts them into Colab's temporary storage.

## Experimental setup

- **Split:** patient-level, 70/15/15 (train/validation/test), stratified by tumor type, seed 42. No patient appears in more than one split; the notebook asserts this. An image-level split with the same proportions is built for the leakage comparison.
- **Preprocessing:** per-image min–max scaling to [0, 1], resize to 224×224. ResNet-18 input is repeated to 3 channels and normalized with ImageNet statistics.
- **Models:** majority-class baseline; 3-block scratch CNN; ResNet-18 pretrained on ImageNet, fully fine-tuned.
- **Training:** cross-entropy loss, Adam (weight decay 1e-4), batch size 16. Scratch CNN: 25 epochs, learning rate 1e-3. ResNet-18: 10 epochs, learning rate 1e-4. The epoch with the best validation macro-F1 is kept and evaluated once on the test set.
- **Reproducibility:** seeds are fixed for Python, NumPy, and PyTorch, with deterministic cuDNN settings.

## Preliminary results (test set)

| Model | Split | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority class | Patient | 0.448 | 0.206 |
| Scratch CNN | Patient | 0.710 | 0.702 |
| ResNet-18 | Patient | 0.895 | 0.888 |
| Scratch CNN | Image | 0.726 | 0.704 |
| ResNet-18 | Image | 0.980 | 0.978 |

Per-class recall, training curves, confusion matrices, and example predictions are in the notebook outputs and in our midterm report.

## Environment

Results above were produced on Google Colab with an NVIDIA T4 GPU, PyTorch 2.11.0 and torchvision 0.26.0.

#
