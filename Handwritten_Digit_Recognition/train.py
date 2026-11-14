
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

print("Loading MNIST dataset...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Split training data into training and validation sets
x_val = x_train[-6000:]
y_val = y_train[-6000:]

x_train = x_train[:-6000]
y_train = y_train[:-6000]

# Resize and normalize images
def preprocess(image, label):

    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=-1)

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        method="bilinear"
    )

    image = image / 255.0

    return image, label


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_dataset = train_dataset.shuffle(10000)

train_dataset = train_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (x_val, y_val)
)

val_dataset = val_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)
)

test_dataset = test_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


print("Building 128x128 CNN model...")

# Build CNN model
model = keras.Sequential([

    layers.Input(shape=(128, 128, 1)),

    # Data augmentation
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),

    # Convolution block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    # Output: digits 0-9
    layers.Dense(10, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()

print("Training 128x128 AI model...")

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)


# Evaluate model
print("Evaluating model...")

loss, accuracy = model.evaluate(test_dataset)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save trained model
model.save("digit_model.keras")

print("128x128 model saved successfully!")
