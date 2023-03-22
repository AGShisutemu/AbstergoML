# Import required libraries
# import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNetV2


class ImageClassifierTrainer:
    def __init__(
        self,
        model_name,
        img_size=224,
        num_classes=4,
        train_dir="../dataset/train",
        val_dir="../dataset/validation",
    ):
        # Define the input size and number of classes
        self.img_size = img_size
        self.num_classes = num_classes

        # Specify the pre-trained model to use
        if model_name == "VGG19":
            self.pretrained_model = VGG19(
                weights="imagenet",
                include_top=False,
                input_shape=(img_size, img_size, 3),
            )
        elif model_name == "ResNet50":
            self.pretrained_model = ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=(img_size, img_size, 3),
            )
        elif model_name == "InceptionV3":
            self.pretrained_model = InceptionV3(
                weights="imagenet",
                include_top=False,
                input_shape=(img_size, img_size, 3),
            )
        elif model_name == "MobileNetV2":
            self.pretrained_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(img_size, img_size, 3),
            )
        else:
            raise ValueError(
                "Invalid model name. " +
                "Supported models are" +
                "VGG19, ResNet50, InceptionV3, and MobileNetV2."
            )

        # Freeze the layers in the pre-trained model
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        # Add a new classifier on top
        x = self.pretrained_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        # Define the new model
        self.model = tf.keras.models.Model(
            inputs=self.pretrained_model.input, outputs=predictions
        )

        # Compile the model
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=True
        )

        # Define the data generators for training and validation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode="categorical",
        )

        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode="categorical",
        )

    def train(self, epochs):
        # Train the model
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples//epochs,
            epochs=10,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples//epochs,
        )

    def save_model(self, model_filename):
        # Save the trained model
        self.model.save(model_filename)
