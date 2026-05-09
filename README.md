# Industrial Conveyor State Classification Pipeline

Computer vision pipeline for industrial conveyor monitoring using TensorFlow and OpenCV.

The system captures images from a conveyor camera, preprocesses the frames, and performs real-time classification using a MobileNetV2-based neural network.

Detected states:
- Empty
- Pallet
- Pallet and Block

---

# Overview

The goal of this project was to develop and validate a lightweight image classification pipeline suitable for industrial and edge-device environments.

The pipeline includes:
- image acquisition from a live camera
- image preprocessing
- transfer learning with MobileNetV2
- real-time inference
- live OpenCV visualization

The project was developed in Python 3.10 using TensorFlow and OpenCV.

---

# Dataset

Before training, a custom dataset was collected and manually separated into three classes:

| Class | Images |
|---|---|
| Empty | 161 |
| Pallet | 159 |
| Pallet and Block | 203 |

Images were captured on the same conveyor system under varying lighting conditions:
- additional shadows
- flashlight illumination
- variable object positioning

This was done to improve robustness under non-ideal industrial conditions.

![Original dataset](doc/images/origin_dataset_size.png)

![Raw image sample](doc/images/raw.png)

---

# Model Selection

The dataset size was too small for training a deep neural network from scratch. For this reason, transfer learning was used.

MobileNetV2 was selected because:
- low computational requirements
- small model size
- suitability for embedded and edge devices
- good performance for lightweight computer vision tasks

---

# Image Preprocessing

The preprocessing pipeline includes:
- center cropping to square aspect ratio
- RGB conversion
- resizing to 224×224 pixels
- MobileNetV2 preprocessing normalization

The preprocessing stage ensures compatibility with the neural network input requirements.

![Preprocessed dataset](doc/images/preprocessed_dataset.png)

Example preprocessing logic:

```python
img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))

img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
img_array = tf.expand_dims(img_array, 0)

img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
```

---
# Transfer Learning

The pretrained MobileNetV2 backbone was used without the original classification head.

Additional layers were added for the target classes:

- GlobalAveragePooling
- Dropout
- Dense softmax classification layer

To improve generalization on the small dataset, lightweight augmentation was added:

- random rotation
- random zoom

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
])
```
---
# Training

The model was trained using:

- Adam optimizer
- learning rate scheduling
- early stopping callbacks

Training parameters:

- batch size: 32
- validation split: 20%
- input size: 224×224
- epochs: 20

The training process converged successfully and early stopping terminated training automatically near epoch 19.

![Accuracy](doc/images/accuracy.png)

---
# Validation Results

The trained model achieved high validation accuracy on the collected dataset.

Evaluation included:

- validation accuracy
- confusion matrix
- visual prediction inspection

![confusion matrix](doc/images/confusion_matrix.png)

![Trained prediction](doc/images/trained_prediction.png)

The model performs well under typical operating conditions and remains relatively stable under moderate lighting variations.

However, some classification errors still occur in boundary conditions:

- partially visible pallets
- atypical block positions
- strong occlusions
- objects near image borders

![Mistake 1](doc/images/mistake_pb1.png)

Mislabel PalletAndBlock

![Mistake 2](doc/images/mistake2.png)

Mislabel Pallet

---
# Fine Tuning

After transfer learning, optional fine tuning was performed by unfreezing the final layers of MobileNetV2.

A lower learning rate was used during fine tuning to avoid destabilizing pretrained weights.

This stage provided minor improvements in validation accuracy.

![Fine tuning results](doc/images/fine_tuning.png)
---
# Real-Time Inference Pipeline

The final pipeline performs:

1. image acquisition from camera
2. frame preprocessing
3. neural network inference
4. confidence estimation
5. OpenCV visualization

Pipeline structure:
```
Camera
  ↓
Frame Capture
  ↓
Preprocessing
  ↓
MobileNetV2 Inference
  ↓
Classification
  ↓
Live Visualization
```

The system samples frames periodically and overlays prediction results directly onto the live camera stream.

![Empty](doc/images/e1.png)

Live pipeline screenshot

![Preprocessed Frame](doc/images/prepocessed_frame.png)

Processed frame screenshot

---
# Results

The developed pipeline successfully classifies conveyor states in real time.

Observed confidence values typically range between:

- 95%–99% in stable conditions
- 50%–90% in boundary conditions or during movement

The lightweight model size makes deployment feasible on:

- embedded systems
- industrial edge devices
- low-power hardware
---
# Improvement Points

Possible future improvements:

- increase dataset size
- improve robustness against lighting changes
- collect additional edge-case samples
- evaluate on a larger independent test dataset
- optimize inference speed for embedded deployment
- add temporal smoothing between predictions
---

# Technologies

- Python 3.10
- TensorFlow
- OpenCV
- NumPy
- MobileNetV2