# Plant Disease Detection: Classical ML vs Deep Learning

Comparative study on the **PlantVillage** dataset, evaluating classical machine learning (HOG + Random Forest/SVM) against deep learning (custom CNN and transfer-learned ResNet50) for multi-class plant leaf disease classification.

## Problem

Identify plant leaf diseases from images,  an agricultural challenge with direct impact on food security and crop yield management.

## Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Original paper:** [Using Deep Learning for Image-Based Plant Disease Detection (2016)](https://arxiv.org/abs/1604.03169)
- **Raw images:** [spMohanty/PlantVillage-Dataset (GitHub)](https://github.com/spMohanty/PlantVillage-Dataset)
- **Size:** 54,305 RGB images across 38 disease/health classes spanning 14 crop species
- **Variant used:** `color/` (RGB, no segmentation masks), one subfolder per class (`<Crop>___<Disease>`)
- **Class imbalance:** 36× between the largest class (`Orange___Haunglongbing_(Citrus_greening)`, 5,507 images) and the smallest (`Potato___healthy`, 152 images)
- **Split:** stratified 70% train / 15% validation / 15% test

In the notebook, the dataset is read from Google Drive (`Summartive_intro_to_ml/color/`) and copied once to the Colab local SSD to avoid Drive I/O latency during training.

## Frameworks

- **Scikit-learn** : classical ML (Random Forest, SVM)
- **TensorFlow/Keras Sequential API** : custom CNN
- **TensorFlow/Keras Functional API** : ResNet50 transfer learning
- **tf.data** : input pipeline (decode, resize, cache, augment, normalize)
- **scikit-image**:  HOG feature extraction

## Experiments

| # | Paradigm | Model | Key Variation | Test Acc |
|---|---|---|---|---|
| 1 | Classical ML | Random Forest | HOG features, baseline | 0.469 |
| 2 | Classical ML | SVM (C=1.0, RBF) | HOG + PCA features | 0.661 |
| 3 | Classical ML | SVM (C=10.0, RBF) | Hyperparameter tuning | 0.668 |
| 4 | TF Sequential | Custom CNN | No augmentation, baseline | 0.906 |
| 5 | TF Sequential | Custom CNN | Data augmentation added | 0.962 |
| 6 | TF Sequential | Custom CNN | Dropout + L2 regularization | 0.845 |
| 7 | TF Functional | ResNet50 | Frozen backbone (transfer learning) | 0.96+ |
| 8 | TF Functional | ResNet50 | Fine-tuned last 30 layers | **0.984** |

The fine-tuned ResNet50 (Exp 8) is the best-performing model, with a macro-average ROC AUC of **0.9998** vs **0.9799** for the best classical model (SVM, C=10).

## Notebook structure (`summartive.ipynb`)

0. Environment setup (GPU check, mixed precision, Drive mount)
1. Exploratory data analysis (class distribution, sample images)
2. Data preprocessing (stratified split, tf.data pipeline, HOG + PCA features)
3. Experiment framework (shared logger, class weights, callbacks)
4. Classical ML experiments (Random Forest, SVM)
5. TF Sequential  custom CNN experiments
6. TF Functional  ResNet50 transfer learning experiments
7. Comprehensive results table
8. Confusion matrices (classical vs deep learning)
9. Classification reports
10. ROC / AUC analysis
11. Learning curves across all DL experiments
12. Deep interpretation & analysis (bias-variance, dataset limitations, proposed improvements)
13. Save best model & results to Drive

## Key design decisions

- **HOG + PCA** for classical ML: reduces 64×64×3 raw pixels to a 1,764-dim HOG descriptor, then projects to 150 PCA components, since disease lesions are characterized by distinctive texture/shape patterns.
- **Subsampling** classical ML to 200 images/class (train) and 60/class (test) keeps SVM training tractable.
- **ImageNet normalization** for deep learning inputs trained from scratch; **caffe-style preprocessing** for ResNet50 to match its pretrained weights.
- **Augmentation** (flip, brightness, contrast, saturation, hue) applied only during training.
- **Class weighting** (`balanced`) used throughout to partially offset the 36× class imbalance.
- **Mixed precision (`mixed_float16`)** for ~1.5–2× speedup on T4 GPU.

## Results & artifacts

Generated plots are in [images/](images/):
- `eda_class_dist.png`, `eda_sample_images.png`  exploratory analysis
- `pca_variance.png`  HOG/PCA explained variance
- `exp4_curves.png` … `exp8_curves.png`  per-experiment accuracy/loss curves
- `all_learning_curves.png`  combined learning curves across all DL experiments
- `results_bar_chart.png`  test accuracy comparison across all 8 experiments
- `cm_classical.png`, `cm_dl.png`  confusion matrices for the best classical and DL models
- `roc_classical.png`, `roc_dl.png`  ROC curves (macro-average, one-vs-rest)

Full numeric results are saved to `experiment_results.csv`; trained models (`.keras`) are saved per experiment, with the best model at `resnet50_finetuned.keras`.

## Running the notebook

1. Open `summartive.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Mount Google Drive and place the PlantVillage `color/` folder at `My Drive/Summartive_intro_to_ml/color/`.
4. Run cells top to bottom  the dataset is copied to local SSD once, then all 8 experiments run sequentially, saving plots/models/results back to Drive.

## Limitations & proposed improvements

- Severe (36×) class imbalance is only partially addressed via class weighting; class-conditional augmentation or embedding-level oversampling could help minority classes.
- Images are lab-quality (uniform background, controlled lighting); real-world field images would need domain adaptation.
- Each image assumes a single disease label; co-occurring diseases on one leaf aren't modeled.
- Next steps: targeted augmentation for the weakest classes, EfficientNetV2/ViT backbones, test-time augmentation, focal loss, and a multi-label architecture for co-occurring diseases.
