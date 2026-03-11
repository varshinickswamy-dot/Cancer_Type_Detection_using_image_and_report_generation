import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
BALANCED_DIR = os.path.join(BASE_DIR, "balanced_dataset")

random.seed(42)

def balance_category(category_name):
    print(f"\nBalancing {category_name}...")

    source_path = os.path.join(DATASET_DIR, category_name)
    target_path = os.path.join(BALANCED_DIR, category_name)

    os.makedirs(target_path, exist_ok=True)

    classes = os.listdir(source_path)

    class_files = {}
    max_count = 0

    # Count files
    for cls in classes:
        cls_path = os.path.join(source_path, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        class_files[cls] = files
        max_count = max(max_count, len(files))

    print("Original counts:")
    for cls in class_files:
        print(f"{cls}: {len(class_files[cls])}")

    # Create balanced dataset
    for cls in class_files:
        src_cls_path = os.path.join(source_path, cls)
        tgt_cls_path = os.path.join(target_path, cls)

        os.makedirs(tgt_cls_path, exist_ok=True)

        files = class_files[cls]
        count = len(files)

        # Copy existing files
        for f in files:
            shutil.copy(os.path.join(src_cls_path, f),
                        os.path.join(tgt_cls_path, f))

        # Oversample if needed
        while count < max_count:
            f = random.choice(files)
            new_name = f"dup_{count}_{f}"
            shutil.copy(os.path.join(src_cls_path, f),
                        os.path.join(tgt_cls_path, new_name))
            count += 1

    print("Balanced successfully.")

# Balance all three
for category in ["breast", "lung", "skin"]:
    balance_category(category)

print("\nAll datasets balanced successfully.")