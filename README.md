# Mangrove Pixel-Level Classification Framework

This is a machine learning/deep learning code repository for mangrove pixel-level classification and feature extraction. The framework integrates data preprocessing, multi-model training, prediction/inference, and spatial morphology-based post-processing (water-body adjacency filtering).

## Core Features

* **Multi-dimensional Spectral Feature Engineering**: Automatically extracts satellite image bands (B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12). Based on the original bands, it automatically calculates spectral and ecological indices such as `NDVI`, `EVI`, `SAVI`, `MNDWI`, and `NDMI` to enhance input features.
* **Comprehensive Model Library**:
    * Supports foundational deep learning models: `LSTM`, `Transformer`, `CNN`, `GRU`, `RNN`, and `MLP`.
    * Supports hybrid deep learning architectures: `CNNLSTM`, `CNNGRU`, and `CNNTransformer`.
    * Includes a custom `ImprovedFixedAdaptiveTransformerV3` model, which combines an `ECA` (Efficient Channel Attention) module with a `GEGLU` activation mechanism.
* **Spatially-Isolated Pre-training Support**: Before model training (`train_DL`), the framework supports loading strictly spatially-isolated pre-trained weights (`pretrain_exclude_{area}_{name}.pth`) to improve generalization for few-shot or cross-regional tasks.
* **Spatial Morphology Post-processing (Water-Adjacency Filtering)**: After the model outputs pixel-level predictions, it uses connected component analysis to extract the mangrove mask (label 1). It then filters this using a water-body mask (processed with a 3x3 dilation kernel). Ultimately, only mangrove connected components adjacent to water bodies are retained, ensuring the output aligns with the true near-water ecological characteristics of mangroves.
* **Visualization Overlay**: Provides a `predict_DL2` method to overlay the post-processed filtered mangrove prediction results in red (`[0, 0, 255]`) with semi-transparency (alpha = 0.5) onto the original RGB background image for intuitive evaluation.

## Training Pipeline

The framework supports two training pipelines: Deep Learning (PyTorch) and Traditional Machine Learning (Scikit-learn).

### Deep Learning Model Training (`train_DL`)
* Uses the `AdamW` optimizer with `weight_decay=0.01` for network parameter updates.
* Performs classification training based on the `CrossEntropyLoss` function.
* Automatically calculates and logs multiple evaluation metrics during training: Accuracy, Precision, Recall, F1-score, and Cohen's Kappa coefficient.
* Automatically saves the optimal model weights (`.pth`) based on the highest validation F1-score (`best_val_f1`).
* Training logs for all epochs are aggregated and exported as a CSV file.

### Traditional Machine Learning Model Training (`train_ML`)
* Provides standard `fit` and `predict` interfaces for model training and inference.
* Similarly calculates Accuracy, Precision, Recall, F1-score, and Kappa coefficient for performance evaluation.
* The final trained model weights are serialized via `joblib` and saved as `.pkl` files.

## Prediction and Inference

The prediction pipeline (`predict_DL`) executes the following steps:
1.  **Data Flattening and Feature Construction**: Flattens the input image matrix, automatically calculates missing spectral indices (replacing NaN/Inf values with the mean), and converts the data into tensors after normalization (via a Scaler).
2.  **Batch Inference**: Uses PyTorch's `DataLoader` to process the data in batches, feeds it into the model to obtain predicted probabilities, and extracts category labels via `argmax`.
3.  **Spatial Filtering**: Loads the environmental water-body mask, dilates the water regions, and filters out all predicted mangrove patches that do not intersect with the dilated water pixels.
4.  **Result Saving**: Outputs both the raw prediction map without post-processing (`_predict_{name}.png`) and the filtered binarized mask map (`_predict_filtered_{name}.png`).
