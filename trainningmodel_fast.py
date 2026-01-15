import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# CONFIGURATION (FAST MODE)
# -------------------------------
DATASET_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\processed_dataset"
IMG_SIZE = (64, 64)
BATCH_SIZE = 16          # increased
EPOCHS = 3               # reduced
STEPS_PER_EPOCH = 4000   # <<< key speed control
VAL_STEPS = 1000
MODEL_SAVE_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\modi_model.keras"

print("Starting FAST training pipeline...")

# -------------------------------
# GPU CONFIG (safe even if no GPU)
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------------
# DATA GENERATOR
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"Detected {NUM_CLASSES} classes")

# -------------------------------
# MODEL
# -------------------------------
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# FAST TRAINING
# -------------------------------
print("FAST training started...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save(MODEL_SAVE_PATH)
print("Model saved at:", MODEL_SAVE_PATH)

# -------------------------------
# PLOTS
# -------------------------------
plt.figure()
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.savefig("accuracy_fast.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.savefig("loss_fast.png")
plt.close()

print("FAST training complete.")
