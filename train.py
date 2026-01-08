import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Config
IMG_SIZE = 224
BATCH_SIZE = 16
DATA_DIR = "dataset/train" # Structure: dataset/train/Healthy, dataset/train/Rust...

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = datagen.flow_from_directory(DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                                        batch_size=BATCH_SIZE, class_mode="categorical", subset="training")

val_data = datagen.flow_from_directory(DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                                      batch_size=BATCH_SIZE, class_mode="categorical", subset="validation")

# Model Architecture
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True # Unfreeze for high accuracy

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
              loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks to reach 95%
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)
]

print("Starting Training...")
model.fit(train_data, validation_data=val_data, epochs=30, callbacks=callbacks)

os.makedirs("model", exist_ok=True)
model.save("model/efficientnetb0_mulberry.keras")
print("✅ Training Complete. Model Saved.")