import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\modi_model2.keras"
IMAGE_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\modi.jpg"

IMG_SIZE = (64, 64)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# LOAD & PREPROCESS IMAGE
# -------------------------------
img = image.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale"
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 64, 64, 1)

# -------------------------------
# PREDICT INDEX
# -------------------------------
predictions = model.predict(img_array)
predicted_index = int(np.argmax(predictions))

print("Predicted class index:", predicted_index)
