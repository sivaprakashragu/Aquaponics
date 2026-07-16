import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Dataset path
DATASET_PATH = "dataset"

# ✅ Balanced augmentation (NOT too aggressive)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

print("Class indices:", train.class_indices)

# ✅ EfficientNet (better than MobileNet)
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

# 🔥 Freeze MOST layers (important)
for layer in base_model.layers[:-40]:
    layer.trainable = False

for layer in base_model.layers[-40:]:
    layer.trainable = True

# Custom head (simplified = better generalization)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

output = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ✅ Better optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Callbacks (important)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
]

# 🚀 Train
model.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=callbacks
)

# Save final model
model.save("fish_model.h5")

print("🔥 Final model trained successfully!")