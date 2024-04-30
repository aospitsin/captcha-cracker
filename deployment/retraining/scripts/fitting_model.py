import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.mobilenet_v2 as mobilenet

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_compiled_model(dropout=0.2, lr=1e-4):
    import tensorflow as tf
    import tensorflow.keras.applications.efficientnet as eff_net
    
    base_model = mobilenet.MobileNetV2(include_top=False,
                                       weights="imagenet",
                                       input_shape=IMAGE_SIZE)
    base_model.trainable = False

    model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    return model

# The path to the folder where the folder with images is located
path_base = r"D:\re-learning" 

file_source = Path(path_base) / "base"
train_file_destination = Path(path_base) / "data" / "train"
val_file_destination = Path(path_base) / "data" / "validation"

gen_data = ImageDataGenerator(
    preprocessing_function=mobilenet.preprocess_input,
    height_shift_range=0.02,
    width_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.15,
    fill_mode="nearest"
)

BATCH_SIZE = 16
IMG_SHAPE = 128
IMAGE_SIZE = (128, 128, 3)

train_dir = train_file_destination
val_dir = val_file_destination

train_data_gen = gen_data.flow_from_directory(
    directory=train_dir, target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=BATCH_SIZE, shuffle=True, class_mode="sparse"
)

val_data_gen = gen_data.flow_from_directory(
    directory=val_dir, target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=BATCH_SIZE, shuffle=True, class_mode="sparse"
)

print("Generators are initialized...")

EPOCHS = 50
NUM_CLASSES = len(train_data_gen.class_indices)
CHECKPOINT_PATH = r"model/weights.hdf5"

model = get_compiled_model(dropout=0.5, lr=1.5e-4)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH, monitor="val_accuracy", verbose=1,
    save_best_only=True, mode="max"
)

print("Model is compiled...\nStart fit...")

history = model.fit(
    train_data_gen, validation_data=val_data_gen,
    epochs=EPOCHS, callbacks=([checkpoint])
)

print(f"Training completed:\nVal_accuracy: {max(history.history['val_accuracy'])}")