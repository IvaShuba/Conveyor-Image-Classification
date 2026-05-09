# Import necessary libraries and check TensorFlow version and GPU availability
import pathlib
import random
import cv2

import matplotlib.pyplot as plt # Plotting library
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
print('TensorFlow version:', tf.__version__)

datapath='./lab_dataset/'


# Define the path to the dataset and set parameters for loading and preprocessing the data

batch_size = 32 
target_size = (224,224) # according to the model requirements
label_mode = "int"
color = "rgb" # according to the model requirements
shuffle = True
validation_split = 0.2 # 20% of the data will be used for validation


train, validation = tf.keras.preprocessing.image_dataset_from_directory(
    datapath,
    validation_split=validation_split,
    subset="both",
    label_mode=label_mode,
    color_mode=color,
    shuffle=shuffle,
    seed=123,
    image_size=target_size,
    batch_size=batch_size,
    crop_to_aspect_ratio = True,
)

# Display class names and dataset sizes
print(train.class_names)
print(len(train.class_names))
print(len(train))
print(len(validation))

# Visualize some sample images from the training dataset
plt.figure(figsize=(30,30), tight_layout=True, facecolor="white")
for image, label in train.take(1):
    for i in range(32):
        ax=plt.subplot(8,4, i+1)
        plt.imshow(image[i].numpy().astype("uint8"), cmap='gray')
        plt.title(train.class_names[label[i]], fontsize=25)
        plt.axis("off")

imagenet_model = tf.keras.applications.MobileNetV2(
    input_shape=target_size + (3,),
    include_top=True,
    weights='imagenet'
)

# Check pre-trained model on a several images from the dataset
sample_images, sample_labels = next(iter(train.take(1)))
sample_images_np = sample_images.numpy()
imagenet_inputs = tf.keras.applications.mobilenet_v2.preprocess_input(sample_images_np.copy())
imagenet_predictions = imagenet_model.predict(imagenet_inputs, verbose=0)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(imagenet_predictions, top=3)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08), # Rotate images by a random angle between -8% and +8%
    tf.keras.layers.RandomZoom(0.10),  # Zoom images by a random from 0% to 10%
], name='data_augmentation')

base_model = tf.keras.applications.MobileNetV2( # Load the MobileNetV2 model without the top classification layer
    input_shape=target_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([# Build a new model by adding custom layers on top of the pre-trained base model
    tf.keras.layers.Input(shape=target_size + (3,), name='input_layer'),
    data_augmentation,
    tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input, name='preprocess_input'),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(train.class_names), activation='softmax')
], name='epb_model') 

model.summary()

# Compile the model with appropriate loss function, optimizer, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Callbacks for learning rate reduction
learning_finished = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.5,
    verbose=1
)

# Callbacks for early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    min_delta=0.001,
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(train,
                    epochs=20,
                    validation_data=validation,
                    callbacks=[early_stop, learning_finished]
                    )

validation_loss, validation_acc = model.evaluate(validation, verbose=2)

print(f'Validation accuracy after transfer learning: {validation_acc:.4f}')

# Save the model
model.save("model_epb_shuba.keras")
# Save weights
model.save_weights("model_epb_shuba_weights.h5")

# Training and validation accuracy over epochs on the plot
plt.plot(history.history['accuracy'], ".b")
plt.plot(history.history['val_accuracy'], ".r")
plt.title('Model accuracy vs Validated accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validated accuracy'], loc='upper left')
plt.ylim([0,1])
plt.show()

#Test model on validation dataset
test_loss, test_acc = model.evaluate(validation, verbose=2)

print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

predictions = model.predict(validation, verbose=0)
y_true = np.concatenate([labels.numpy() for _, labels in validation], axis=0)
y_pred = np.argmax(predictions, axis=1)

print('Classification report:')
print(classification_report(y_true, y_pred, target_names=validation.class_names))

#Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion matrix on the test set')
plt.xticks(range(len(validation.class_names)), validation.class_names, rotation=20)
plt.yticks(range(len(validation.class_names)), validation.class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()

#Visualize predictions on validation dataset
plt.figure(figsize=(16, 36))
class_names = validation.class_names
for images, labels in validation.take(1):
    for i in range(32):
        result = "Correct" if labels[i] == np.argmax(predictions[i]) else "MISLABEL"
        ax = plt.subplot(8, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
        plt.title(  "#" + str(i+1) + " " + result +
                    "\n" + class_names[labels[i]] + " / " + class_names[np.argmax(predictions[i])] + " | (Original/Prediction)"
                    +"\nCertainty: " + str(round(max(predictions[i])*100.,2)) + "%"
                    )
        plt.axis("off")

plt.savefig("result.png")