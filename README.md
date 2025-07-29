# Chest X‑Ray Pneumonia Classification

This repository contains two Jupyter notebooks that demonstrate how to train and evaluate deep learning models for detecting pneumonia in chest X‑ray images. In medical imaging, pneumonia presents as abnormal opacities in the lungs, and automatic detection can support radiologists in making faster and more consistent diagnoses. Both notebooks rely on transfer learning using a pre‑trained DenseNet121 convolutional neural network to classify X‑rays as either normal or pneumonia.

## Datasets

Two different publicly available datasets are used in these notebooks:

### Chest X‑Ray Images (Pneumonia)
This pediatric dataset contains 5,863 JPEG images of chest X‑rays organized into separate training, test and validation folders. Each folder contains sub‑directories labelled PNEUMONIA and NORMAL. The images were acquired from pediatric patients aged 1–5 at the Guangzhou Women and Children’s Medical Center and were screened for quality by expert radiologists. This dataset is hosted on Kaggle and must be downloaded manually because it is not included in this repository. After downloading, place the `train/`, `test/` and `val/` folders under a `data/chest_xray/` directory so that the notebook can locate them.

### NIH ChestX‑ray14
The National Institutes of Health released a large dataset of 112,120 frontal chest X‑ray images annotated with up to 14 thoracic pathologies. The notebooks use a CSV file (`labeled_images.csv`) that contains the paths to selected images and their associated labels. Only two classes are considered: images labeled *No Finding* are mapped to NORMAL and images where the string “Pneumonia” appears in the findings are mapped to PNEUMONIA. To balance the classes, the notebook downsamples the normal images to match the number of pneumonia cases. Users must download the images and CSV file from the NIH dataset and adjust the paths accordingly.

## What is DenseNet?

Both notebooks fine‑tune a DenseNet121 model. DenseNet is a deep learning architecture in which each layer receives input from all preceding layers and passes its output to all subsequent layers. This dense connectivity pattern alleviates the vanishing‑gradient problem and improves feature propagation. DenseNet models reuse features, which makes them parameter‑efficient compared with traditional convolutional networks. The 121‑layer variant used here has been pre‑trained on ImageNet and provides a strong starting point for medical imaging tasks.

## Notebook Overview

### `ChexNetFinal (2).ipynb`
This notebook demonstrates a complete training pipeline for the pediatric Chest X‑Ray Images dataset.

- **Data loading and preprocessing** – The `image_dataset_from_directory` API reads images from the `train/`, `val/` and `test/` folders and resizes them to 224×224 pixels. Images are normalized by scaling pixel intensities to the [0, 1] range.
- **Model definition** – A Keras Sequential model is created that incorporates a frozen DenseNet121 base, a global average pooling layer, a fully connected layer with 128 units and ReLU activation, a 50% dropout layer to reduce overfitting, and a final sigmoid unit for binary classification.
- **Class balancing and training** – Class weights are computed to address dataset imbalance. The model is trained using the Adam optimizer and binary cross‑entropy loss with early stopping.
- **Evaluation** – Includes confusion matrix, classification report, and AUC.
- **Single‑image prediction and Grad‑CAM** – The model makes predictions and visualizes key regions with Grad‑CAM.

### `NIHbinaryclassification (1).ipynb`
This notebook targets binary classification on the NIH ChestX‑ray14 dataset.

- **Data preparation** – Uses a CSV with paths and labels. Normal cases are downsampled to balance the dataset. Stratified train/validation splits preserve class distribution.
- **Dataset pipeline** – `tf.data.Dataset` is used with image resizing and normalization.
- **Model definition and fine‑tuning** – Uses DenseNet121 with global average pooling, dense layer (128 units, L2 regularization), dropout, and sigmoid output. Class weights and learning-rate scheduler are included.
- **Evaluation and visualisation** – Includes classification report, confusion matrix, Grad‑CAM visualization, and ROC curve.

## Running the Notebooks

- **Install dependencies** – Create a Python environment and run:

  ```bash
  pip install tensorflow pandas numpy scikit-learn matplotlib seaborn opencv-python
  ```

- **Download datasets** – Manually download and extract datasets from Kaggle and the NIH site. Update paths in the notebooks as needed.
- **Launch Jupyter** – Run `jupyter notebook` in the repo root and execute cells sequentially. GPU is highly recommended.
- **Model saving/loading** – Trained models are saved in the `saved_models/` directory in `.keras` format.

## Customisation

- **Adjusting image size** – Modify the `image_size` argument in the dataset loading functions.
- **Changing network depth** – Try other models like DenseNet169 or EfficientNet.
- **Handling class imbalance** – Try oversampling or using focal loss instead of downsampling.
- **Extending to multi‑label classification** – Replace the final sigmoid unit with a multi‑output layer for detecting multiple diseases using binary cross‑entropy.

## Disclaimer

These notebooks are educational examples and should not be used for diagnostic purposes. They demonstrate how to prepare data, fine‑tune a pre‑trained network, and evaluate its performance. Real‑world clinical use requires rigorous validation, regulatory approval and expert oversight.
