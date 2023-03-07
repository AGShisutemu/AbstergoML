import numpy as np
import keras.utils as image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19


class VGG19CoffeeClassifier:
    def __init__(self, pretrained=True, model_path=None):
        if pretrained:
            # Load the pre-trained VGG19 model without the top layers
            self.base_model = VGG19(weights='imagenet', include_top=False)

            # Add new classification layers to the model
            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(4, activation='softmax')(x)

            # Define the new model with the added classification layers
            self.model = Model(inputs=self.base_model.input, outputs=predictions)

            # Freeze the weights of the pre-trained layers
            for layer in self.base_model.layers:
                layer.trainable = False

            # Compile the model with a SGD optimizer and categorical crossentropy loss
            self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # Load the custom model from the .h5 file
            self.model = load_model(model_path)

        # Define the image size for the model input
        self.img_width, self.img_height = 224, 224

        # Define the label dictionary
        self.label_dict = {0: "Unripe", 1: "Semi-ripe", 2: "Ripe", 3: "Overripe"}

    def classify(self, image_path):
        # Load the image to classify
        img = image.load_img(image_path, target_size=(self.img_width, self.img_height))

        # Convert the image to an array and preprocess it for input into the model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make a prediction with the model
        preds = self.model.predict(x)

        # Decode the predictions and return the predicted label
        pred_index = np.argmax(preds)
        pred_label = self.label_dict[pred_index]
        accuracy = preds[0][pred_index]
        return {"accuracy": f"{accuracy:.2f}", "label": pred_label}


# Usage
# vgg_coffee = VGG19CoffeeClassifier(pretrained=True, model_path='my_model.h5')
# image_path = 'coffee_berry.png'
# pred_label = vgg_coffee.classify(image_path)
# print('Predicted:', pred_label)