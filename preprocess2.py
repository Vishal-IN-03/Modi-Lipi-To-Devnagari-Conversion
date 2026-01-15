import cv2
import os


input_dir = r"C:\Users\pvish\OneDrive\Documents\modi-dev\dataset\Modi Lipi\MODI_HChar\MODI_HChar"
output_dir = r"C:\Users\pvish\OneDrive\Documents\modi-dev\processed_dataset"


os.makedirs(output_dir, exist_ok=True)
print(" Output folder ready:", output_dir)


if not os.path.exists(input_dir):
    print(" Input folder does not exist! Check path:", input_dir)
    exit()

print(" Input folder found:", input_dir)


for root, dirs, files in os.walk(input_dir):
    
    if not files:
        continue

    
    rel_path = os.path.relpath(root, input_dir)
    save_path = os.path.join(output_dir, rel_path)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n Processing folder: {rel_path} ({len(files)} files)")

    for file in files:
        img_path = os.path.join(root, file)

        
        img = cv2.imread(img_path)
        if img is None:
            print(f" Could not read {file}")
            continue

        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(binary, (64, 64))

        
        out_path = os.path.join(save_path, file)
        cv2.imwrite(out_path, resized)
        print(f" Saved: {out_path}")

print("\n All preprocessing complete! Clean images saved in:", output_dir)
