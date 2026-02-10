# CLWD: A Chinese Histopathology Dataset for Lung Adenocarcinoma Subtype Classification

This repository contains the official implementation code for the paper **"CLWD: a Chinese histopathology dataset for lung adenocarcinoma subtype classification"**.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## 🔍 Overview

This project provides code for lung adenocarcinoma subtype classification using whole slide images (WSIs). We implement and evaluate three state-of-the-art Multiple Instance Learning (MIL) methods on the CLWD dataset.

## 🛠️ Installation

### 1. Create Conda Environment

First, create a conda environment named `clwd` with Python 3.8:

```bash
conda create -n clwd python=3.8
conda activate clwd
```

### 2. Install Dependencies

Install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preprocessing

The preprocessing pipeline is adapted from [vkola-lab/tmi2022](https://github.com/vkola-lab/tmi2022).

### Step 1: Patch Extraction

Use `tile_WSI.py` to extract patches from whole slide images:

```bash
python utils/tile_WSI.py [options] <slide_path>
```

**Key parameters:**
- `-s, --size`: Patch size in pixels (default: 256)
- `-o, --output`: Output directory for patches
- `-B, --Background`: Maximum background threshold percentage (default: 50)
- `-j, --jobs`: Number of worker processes (default: 4)

### Step 2: Feature Extraction

Use `build_graphs.py` to extract features from patches and build graph structures:

```bash
python utils/build_graphs.py --dataset <patch_path> --backbone <model> --weights <weights_path> --output <output_path>
```

**Configuration for different methods:**

- **CLAM & TransMIL:**
  - Patch size: 256×256
  - Backbone: ResNet50 (pretrained on ImageNet)
  
- **GraphTransformer:**
  - Patch size: 512×512
  - Backbone: ResNet18 (pretrained weights provided in `checkpoints/resnet18.pth`)


## 🚀 Model Training

We provide implementations of three MIL methods. Configuration files are located in the `config/` directory.

### CLAM 

```bash
python CLAM.py
```

### TransMIL

```bash
python TransMIL.py
```

### GraphTransformer

```bash
python GraphTransformer.py
```

**Note:** Modify the configuration files in `config/` to adjust hyperparameters, data paths, and training settings.

## 📈 Model Evaluation

### 1. Plot Performance Curves

Generate ROC curves, PR curves, and confusion matrices for k-fold cross-validation:

```bash
python plot_curves.py \
  --save_dir results/<model_name> \
  --n_folds 5 \
  --class_names papillary lepidic "in situ" solid micropapillary cribriform acinar
```

This will generate:
- `roc_curves_all_classes.png`: ROC curves with AUC scores
- `pr_curves_all_classes.png`: Precision-Recall curves with AP scores
- `confusion_matrix_average.png`: Average confusion matrix across folds
- `confusion_matrix_std.png`: Standard deviation of confusion matrix

### 2. Dataset Feature Similarity Analysis

Explore feature similarity between CLWD and DHMC datasets using UMAP:

```bash
python dataset_umap.py
```

This will generate:
- `umap_2d_visualization.png`: 2D UMAP visualization
- `umap_3d_visualization.png`: 3D UMAP visualization

The visualizations help understand the feature distribution and similarity between different datasets.


## 🙏 Acknowledgments

We gratefully acknowledge the following resources and works that made this project possible:

### Datasets

- **CLWD Dataset**: The dataset is publicly available and can also be accessed directly through our Pathology Image Repository ( https://leelab.kmmu.edu.cn/PathologyRepository ). Otherwise, the JPG version of the dataset also available at the Hugging Face repository ( https://huggingface.co/datasets/kmmuleelab/Lung_Pathology_Image_JPG ).
- **DHMC Dataset**: [Dartmouth Lung Cancer Histology Dataset](https://bmirds.github.io/LungCancer/)

### Methods
This project implements and evaluates the following methods:

1. **CLAM** - Clustering-constrained Attention Multiple Instance Learning
   - Paper: [Data-efficient and weakly supervised computational pathology on whole-slide images](https://www.nature.com/articles/s41551-020-00682-w)
   - Original Code: [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)

2. **TransMIL** - Transformer-based Multiple Instance Learning
   - Paper: [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://proceedings.neurips.cc/paper_files/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)
   - Original Code: [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL)

3. **GraphTransformer** - Graph Transformer for Histopathology
   - Paper: [A graph-transformer for whole slide image classification](https://doi.org/10.1109/TMI.2022.3176598)
   - Original Code: [vkola-lab/tmi2022](https://github.com/vkola-lab/tmi2022)

We also thank the preprocessing pipeline adapted from [vkola-lab/tmi2022](https://github.com/vkola-lab/tmi2022).

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{,
  title={CLWD: a Chinese histopathology dataset for lung adenocarcinoma subtype classification},
  author={},
  journal={},
  year={}
}
```
---

**Note**: Please ensure you have the necessary permissions and comply with data usage agreements when using the CLWD and DHMC datasets.
