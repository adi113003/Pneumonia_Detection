# NIHbinaryclassification Notebook

This notebook trains and evaluates a binary classifier to detect pneumonia in chest X‑ray images from the NIH ChestX‑ray14 dataset. The original dataset contains 112,120 frontal chest X‑ray images collected from over 30,000 patients and annotated with up to 14 pathologies. Because the images cover many thoracic diseases, the notebook filters and balances the data to create a two‑class problem: PNEUMONIA vs NORMAL.

## Data Preparation

### Loading the Label CSV
The notebook expects a CSV file (`labeled_images.csv`) with:
- `image_path`: Full path to each image.
- `label`: Pathology strings.

Labels are mapped to:
- **PNEUMONIA**: if "Pneumonia" is present.
- **NORMAL**: if the entry is "No Finding".
Other labels are discarded.

### Balancing the Classes
To address imbalance, normal cases are randomly downsampled (e.g., to 1,450) to match pneumonia cases. A seed ensures reproducibility.

### Train/Validation Split
The balanced dataset is split 80/20 using stratified sampling. Labels are encoded as:
- 0 = NORMAL
- 1 = PNEUMONIA

### Building a `tf.data` Pipeline
- Converts file paths and labels to TensorFlow Dataset.
- Reads, decodes, resizes (224×224), and normalizes images to [0, 1].
- Includes shuffling, batching, prefetching.
- Computes class weights for residual imbalance.

## Model Architecture and Training

Uses transfer learning with a DenseNet121 base (pre-trained on ImageNet), wrapped in a Keras Sequential model with:

- `GlobalAveragePooling2D` to reduce spatial dimensions.
- Dense layer (128 units, ReLU, L2 regularization).
- Dropout layer (rate 0.5).
- Final dense layer with sigmoid activation (1 unit).

**Compiled with:**
- Adam optimizer (`learning_rate=1e-4`)
- Binary cross-entropy loss
- `EarlyStopping` and `ReduceLROnPlateau` callbacks
- Class weights passed during training

## Evaluation

Evaluated on the validation set with:
- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix**
- **ROC Curve** (with AUC)

Displays random validation samples with true vs. predicted labels for qualitative inspection.

## Grad‑CAM and Misclassification Analysis

Applies **Grad‑CAM** to:
- Highlight regions influencing the model's prediction.
- Check model focus on relevant lung areas.

Displays Grad‑CAM visualizations for both:
- Correct predictions
- Misclassified examples (e.g., due to subtle signs or noise)

## Saving and Loading the Model

The trained model is saved as:

```
saved_models/nih_binary_model_finetuned.keras
```

Load it later with:

```python
tf.keras.models.load_model("saved_models/nih_binary_model_finetuned.keras")
```

## Running the Notebook

1. **Install dependencies:**

    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib opencv-python
    ```

2. **Download NIH Dataset:**
   - Download ChestX‑ray14 images and `labeled_images.csv`
   - Update the `image_path` column in the CSV to match local image locations
   - Optionally extract only NORMAL and PNEUMONIA images for speed

3. **Run the Notebook:**
   - Open `NIHbinaryclassification (1).ipynb` in Jupyter
   - Run cells sequentially (GPU highly recommended)

4. **Inspect Outputs:**
   - Classification report
   - Confusion matrix
   - ROC curve
   - Grad‑CAM heatmaps and visualizations

## Possible Improvements

- **Use more data:** Instead of downsampling, oversample PNEUMONIA or use class-balanced sampling.
- **Data augmentation:** Apply random flips, rotations, or brightness changes.
- **Fine‑tuning:** Unfreeze some DenseNet121 layers and retrain at a lower learning rate.
- **Multi‑label classification:** Predict 14 thoracic diseases by changing the output to 14 sigmoid units and using multi-label loss.

## Disclaimer

This notebook is intended for research and educational use only. It should not be used for clinical decision-making. Always consult licensed medical professionals when interpreting radiographic images.
