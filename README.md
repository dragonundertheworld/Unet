# Road Scene Semantic Segmentation (U‑Net)

This repository implements a semantic segmentation pipeline for road‑scene understanding using a U‑Net architecture in TensorFlow/Keras. It covers data loading, preprocessing (cropping, normalization), model definition, training, evaluation (per‑class accuracy & Mean IoU), and uncertainty quantification (aleatoric & epistemic).

---

## Table of Contents

- [Road Scene Semantic Segmentation (U‑Net)](#road-scene-semantic-segmentation-unet)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
  - [Data Loading \& Preprocessing](#data-loading--preprocessing)
    - [`Dataloader` class](#dataloader-class)
    - [Processing functions](#processing-functions)
    - [TF Dataset pipelines](#tfdataset-pipelines)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Per‑class pixel accuracy](#perclass-pixel-accuracy)
    - [Mean IoU](#mean-iou)
  - [Uncertainty Analysis](#uncertainty-analysis)
  - [Usage](#usage)
  - [License](#license)

---

## Project Structure

```
.
├── docs
├── scripts/CV1-05-lab2.ipynb
└── README.md                  # This file
```

---

## Prerequisites

- Python 3.8+  
- TensorFlow 2.x  
- h5py  
- numpy  
- matplotlib  
- pandas  
- scikit‑learn  

Install with:
```bash
pip install -r requirements.txt
```

---

## Data

We use two HDF5 files each containing:

- `rgb` – raw RGB images as `(N, H, W, 3)` arrays  
- `seg` – per‑pixel integer labels `(N, H, W, 1)`  
- `color_codes` – lookup table for mapping class IDs to RGB colors  

Place `driving_train_data.h5` and `driving_test_data.h5` in the repo root.

---

## Data Loading & Preprocessing

### `Dataloader` class  
Wraps an HDF5 file and provides:
```python
trainset = Dataloader("./driving_train_data.h5")
len(trainset)              # number of examples
img, lbl = trainset.getitem(i)
```

### Processing functions  
- **`process_trainset(trainset, crop=True)`**  
  - Concatenates `rgb` + `seg` → random crop to (128×256).  
- **`process_testset(testset, normal=True)`**  
  - Casts RGB to float32 in `[-1,1]` and labels to `int32`.

### TF Dataset pipelines  
```python
dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
    .cache().repeat().shuffle(300).batch(32).prefetch(tf.data.AUTOTUNE)

dataset_val   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)) \
    .cache().batch(32)
```

---

## Model Architecture

We implement a standard **U‑Net**:

```
Input → [Conv×2 → BN → MaxPool] × 4 → Bottom Conv blocks
→ [UpConv → Concatenate skip → Conv×2 → BN] × 4 
→ 1×1 Conv → Softmax over 34 classes
```

- All conv layers use `3×3` kernels, `padding='same'`, ReLU activation, followed by BatchNorm.  
- Upsampling with `Conv2DTranspose` (strides=2).  
- Final layer: `Conv2D(34, 1, activation='softmax')`.

Instantiate and compile with:
```python
model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc', MeanIoU(num_classes=34)]
)
```

---

## Training

```python
history = model.fit(
    dataset_train,
    epochs=100,
    steps_per_epoch = len(trainset)//32,
    validation_data  = dataset_val,
    validation_steps = len(testset)//32
)
model.save('model.h5')
```

Adjust `BATCH_SIZE`, `EPOCHS`, and `BUFFER_SIZE` at the top of the notebook.

---

## Evaluation

### Per‑class pixel accuracy  
Compute confusion matrices over the test set and plot a bar chart of accuracy for each class (excluding background).

### Mean IoU  
Logged automatically during `model.compile(..., metrics=[MeanIoU(...)])`.

---

## Uncertainty Analysis

We estimate:
- **Aleatoric uncertainty** – pixelwise entropy of the mean softmax over multiple stochastic forward passes.
- **Epistemic uncertainty** – variance (std) across multiple dropout‑enabled predictions.

In the notebook, we:
1. Run `T` stochastic passes with dropout kept active.  
2. Compute per‑pixel variance and entropy maps.  
3. Visualize alongside input, ground truth, and predicted mask.

---

## Usage

1. Open `segmentation.ipynb` in Jupyter.  
2. Modify file paths or hyperparameters as needed.  
3. Run all cells to train, save, and evaluate the model.  

Alternatively, import functions:
```python
from segmentation import Dataloader, process_trainset, create_model
```

---

## License

This code is released under the MIT License. Feel free to adapt for your own research and projects.
