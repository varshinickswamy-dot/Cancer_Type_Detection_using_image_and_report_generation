import os
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CHANGE THIS PER FILE
# ==============================
MODEL_NAME = "lung"   # change to breast / lung
# ==============================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 8      # fast training
EPOCHS_PHASE2 = 8      # fine-tuning

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "balanced_dataset", MODEL_NAME)
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_model.h5")
CLASS_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_classes.json")

# ======================================================
# DATA GENERATORS (CORRECT EFFICIENTNET PREPROCESSING)
# ======================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# ======================================================
# CLASS WEIGHTS (PREVENT BIAS)
# ======================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ======================================================
# MODEL (FAST + ACCURATE)
# ======================================================

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

early_stop = callbacks.EarlyStopping(
    monitor="val_auc",
    patience=3,
    restore_best_weights=True
)

# ==============================
# PHASE 1 TRAINING
# ==============================

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ==============================
# PHASE 2 FINE-TUNING
# ==============================

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ======================================================
# SAVE MODEL
# ======================================================

model.save(MODEL_PATH)

with open(CLASS_PATH, "w") as f:
    json.dump(train_data.class_indices, f)

print(f"{MODEL_NAME.upper()} model trained and saved successfully.")