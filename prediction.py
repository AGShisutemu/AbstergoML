import numpy as np
import tensorflow as tf
import keras.utils as image
from keras.applications.vgg19 import preprocess_input

model = tf.keras.applications.VGG19(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

for layer in model.layers[:-5]:
    layer.trainable = False

x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Assume that the training data is stored in the train_data directory
img = image.load_img("coffee_berry.png", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

ripeness_labels = {0: "Unripe", 1: "Semi-ripe", 2: "Ripe", 3: "Overripe"}
prediction = model.predict(img_array)
predicted_ripeness = ripeness_labels[np.argmax(prediction)]

# Print the predicted ripeness of the coffee berry
print("Predicted ripeness:", predicted_ripeness)
# Get the accuracy in percentage in percent
print("Accuracy:", np.max(prediction) * 100)

# TODO MAKE TKINTER GUI