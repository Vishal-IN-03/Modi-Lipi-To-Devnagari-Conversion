from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\modi_model2.keras"
model = load_model(MODEL_PATH)

print("Model loaded successfully")
