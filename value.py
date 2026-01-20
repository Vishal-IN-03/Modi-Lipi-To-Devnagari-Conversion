import os

# -------------------------------
# ROOT DATASET PATH
# -------------------------------
ROOT_PATH = r"C:\Users\pvish\OneDrive\Documents\modi-dev\training_data"

# -------------------------------
# GET TOP-LEVEL FOLDERS
# -------------------------------
top_folders = sorted([
    f for f in os.listdir(ROOT_PATH)
    if os.path.isdir(os.path.join(ROOT_PATH, f))
])

print("\nHIERARCHICAL NUMERICAL ORDER\n")

for top_idx, top_name in enumerate(top_folders):
    top_path = os.path.join(ROOT_PATH, top_name)
    print(f"{top_idx} → {top_name}")

    # -------------------------------
    # GET SUB-FOLDERS
    # -------------------------------
    sub_folders = sorted([
        sf for sf in os.listdir(top_path)
        if os.path.isdir(os.path.join(top_path, sf))
    ])

    for sub_idx, sub_name in enumerate(sub_folders):
        print(f"   {top_idx}.{sub_idx} → {sub_name}")

# Get mapping from training
class_indices = train_gen.class_indices

# Reverse it: index -> folder name
index_to_folder = {v: k for k, v in class_indices.items()}

# Final mapping: index -> Devanagari
label_map = {
    i: devanagari_mapping[index_to_folder[i]]
    for i in index_to_folder
}
