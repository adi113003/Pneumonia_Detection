# ChexNetFinal Notebook

This notebook demonstrates how to build, train, and interpret a binary classification model for detecting pneumonia from pediatric chest X‑ray images. It draws inspiration from the CheXNet paper released by Stanford, which showed that deep convolutional networks can surpass radiologist performance on pneumonia detection. The notebook leverages a pre‑trained DenseNet121 backbone and performs transfer learning on a small, well‑curated pediatric dataset.

## Dataset

The notebook uses the Chest X‑Ray Images (Pneumonia) dataset, which contains 5,863 JPEG images organized into `train/`, `val/`, and `test/` directories, each with `PNEUMONIA` and `NORMAL` sub‑folders. The images were captured from pediatric patients aged one to five years old and vetted by expert radiologists. Ensure that the dataset is downloaded from Kaggle and extracted into a `data/chest_xray/` directory so that the notebook can locate the files.

## Key Steps

### Data Loading and Preprocessing
- Uses `tf.keras.preprocessing.image_dataset_from_directory` to create training and validation datasets with a 70/30 split.
- Each image is resized to 224×224 pixels and normalized to the [0, 1] range using a `Rescaling` layer.
- The test dataset is loaded from the `test` folder.

### Model Architecture
- A transfer-learning model is built by stacking:
  - A frozen DenseNet121 base pre-trained on ImageNet.
  - `GlobalAveragePooling2D` to convert feature maps into a feature vector.
  - A dense layer with 128 units and ReLU activation.
  - A `Dropout(0.5)` layer.
  - A final dense layer with sigmoid activation for binary classification.

### Training with Class Weights
- Class weights are computed using `sklearn.utils.class_weight` to address class imbalance.
- The model uses the Adam optimizer (learning rate 1e-4), binary cross-entropy loss, and an `EarlyStopping` callback to monitor validation loss.

### Evaluation
- After training (up to 10 epochs), the model is evaluated on the test set.
- It generates a confusion matrix, classification report (precision, recall, F1-score), and ROC curve with AUC.

### Prediction and Grad‑CAM Visualisation
- A sample chest X-ray is pre-processed and classified by the model.
- Grad‑CAM visualizes which regions of the image influenced the decision by producing a heatmap based on gradients from the final convolutional layer.

### Saving the Model
- The model is saved in the `saved_models/` directory using the `.keras` format.
- Reload the model later with `tf.keras.models.load_model()`.

## How to Run

1. **Install required packages:**

    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
    ```

2. **Download and extract dataset:**
    Ensure the structure looks like:

    ```
    data/chest_xray/train/PNEUMONIA
    data/chest_xray/train/NORMAL
    data/chest_xray/test/PNEUMONIA
    data/chest_xray/test/NORMAL
    data/chest_xray/val/PNEUMONIA
    data/chest_xray/val/NORMAL
    ```

3. **Launch Jupyter Notebook:**
    Open `ChexNetFinal.ipynb` and run the cells sequentially. GPU is recommended for faster training.

4. **Evaluate the Model:**
    After training, check the confusion matrix, classification report, and ROC curve to assess performance. Use Grad‑CAM to inspect attention regions in the lungs.

## Suggested Extensions

- **Fine‑tuning:** Unfreeze deeper layers in DenseNet121 and retrain with a lower learning rate for improved accuracy.
- **Data Augmentation:** Add horizontal flips, rotations, or brightness changes to enrich training data.
- **Alternative Architectures:** Try EfficientNet or ResNet50 for comparison.
- **Multi‑class Classification:** Modify the model to predict multiple thoracic diseases using a multi-label output layer.

## Disclaimer

This notebook is intended for research and educational purposes only. It should not be used in clinical decision-making. Always consult licensed medical professionals when interpreting radiographic images.
