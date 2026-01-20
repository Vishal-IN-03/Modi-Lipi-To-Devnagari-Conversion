import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
MODEL_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\modi_model2.keras"
model = load_model(MODEL_PATH)

# -------------------------------
# LABEL MAP (EXACT FOLDER ORDER)
# -------------------------------
label_map = {
    
    0:  "अ",   # a
    1:  "आ",   # aa
    2:  "अः",  # ah
    3:  "ऐ",   # ai
    4:  "अं",  # am

    5:  "ब",   # b
    6:  "भ",   # bh
    7:  "च",   # ch
    8:  "छ",   # chh
    9:  "ढ",   # d
    10: "ध",   # dh
    11: "द",  # dha
    12: "ज्ञ ", # dyn

    13: "ए",   # e
    14: "८",   # eight
    15: "५",   # five
    16: "४",   # four

    17: "ग",   # g
    18: "घ",   # gh
    19: "ह",   # h

    20: "इ",   # i
    21: "ई",   # ii

    22: "ज",   # ja
    23: "झ",   # jh

    24: "क",   # k
    25: "ख",   # kh
    26: "क्ष", # ksh

    27: "ल",   # l
    28: "ळ",   # lh
    29: "म",   # m
    30: "न",   # n

    31: "९",   # nine
    32: "ण",   # nn

    33: "ओ",   # o
    34: "१",   # one

    35: "ऊ",   # oo
    36: "औ",   # ou

    37: "प",   # p
    38: "फ",   # ph
    39: "र",   # r
    40: "स",   # s

    41: "७",   # seven
    42: "श",   # sh
    43: "श्र", # shr
    44: "६",   # six

    45: "त",   # ta
    46: "ट",   # th
    47: "थ",  # tha
    48: "ठ",   # thh
    49: "ध",  # thha

    50: "३",   # three
    51: "त्र", # tr
    52: "२",   # two

    53: "उ",   # u
    54: "व",   # v
    55: "य",   # y
    56: "०"    # zero
}




# -------------------------------
# LOAD TEST IMAGE
# -------------------------------
IMAGE_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\processed_dataset_flat\ch\ch6.jpg"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0
img = img.reshape(1, 64, 64, 1)

# -------------------------------
# PREDICT
# -------------------------------
prediction = model.predict(img)
predicted_class = int(np.argmax(prediction))
predicted_char = label_map[predicted_class]

print("Predicted class index:", predicted_class)
print("Predicted character:", predicted_char)
