# Import necessary libraries and check TensorFlow version and GPU availability
import pathlib
import random
import cv2
import time
import numpy as np

import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
print('TensorFlow version:', tf.__version__)

CLASS_NAMES = [
    "Empty",
    "Pallet",
    "Pallet and Block"
]

print("Loading model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights=None # Because we load our weights
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax') # 3 Classes
])

# applieyng weights 
model.load_weights("model_weights.weights.h5")
print("Weights are loaded TF!")
print("Ready!")


def predict_frame(frame, model, class_names):
    # 1.  crop_to_aspect_ratio=True
    h, w, _ = frame.shape
    min_side = min(h, w)
    start_x = (w - min_side) // 2
    start_y = (h - min_side) // 2
    frame_cropped = frame[start_y:start_y+min_side, start_x:start_x+min_side]

    # 2. RGB + Resize
    img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 3. prepearing array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # (1, 224, 224, 3)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # 5. Prediction
    predictions = model.predict(img_preprocessed, verbose=0)
    
    # best prediction as confidence
    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id]
    cv2.imshow("Processed Frame", img_resized)
    return class_names[class_id], confidence

cap = cv2.VideoCapture(0)
last_prediction_time = 0
prediction_interval = 2.0 # sec
current_label = "Waiting..."
current_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    
    # Sampling every 2 sec
    if current_time - last_prediction_time > prediction_interval:
        
        current_label, current_conf = predict_frame(frame, model, CLASS_NAMES)
        last_prediction_time = current_time
        print(f"Result: {current_label} ({current_conf:.2%})")

    # Visualisation: Show text on cam image
    text = f"Status: {current_label} ({current_conf:.2%})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Lab Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()