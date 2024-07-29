# Solar Panel Condition Classifier

## Overview
This project is aimed at developing a model to recognize the state of solar panels and classify them into one of the following categories: 'Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', and 'Snow-Covered'. The model uses the VGG16 architecture as a base for feature extraction and is fine-tuned to achieve accurate classification results.

## Dataset
The dataset is organized into directories with subdirectories for each class. The images are automatically split into training and validation sets using TensorFlow's `image_dataset_from_directory` utility.

## Model Architecture
The model is based on the VGG16 architecture, pretrained on the ImageNet dataset. The top layers are removed, and a global average pooling layer, dropout layer, and dense layer are added for classification.

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- tensorflow
- opencv-python (cv2)
- glob
- requests

Install the required packages using pip:
```bash
pip install pandas numpy seaborn matplotlib tensorflow opencv-python glob requests
```
## Code Explanation

Import Libraries:
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random
from cv2 import resize
from glob import glob
import requests
from io import BytesIO
```

Load and Prepare Dataset:
```bash
img_height = 244
img_width = 244
train_ds = tf.keras.utils.image_dataset_from_directory(
    './Faulty_solar_panel/Faulty_solar_panel',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    './Faulty_solar_panel/Faulty_solar_panel',
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True
)
```

Display Class Names and Sample Images:
```bash
# Checking the Classes
class_names = train_ds.class_names
print(class_names)

# Displaying Sample Images
plt.figure(figsize=(15, 15))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

```

Model Definition:
```bash
base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False 

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(90)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

Compile and Train the Model:
```bash
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoch = 15
model.fit(train_ds, validation_data=val_ds, epochs=epoch,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-2,
            patience=3,
            verbose=1,
            restore_best_weights=True
        )
    ]
)
```

Fine-Tuning:
```bash
base_model.trainable = True
for layer in base_model.layers[:14]:
    layer.trainable = False
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoch = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=epoch,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-2,
            patience=3,
            verbose=1,
        )
    ]
)
```

Plot Training History:
```bash
get_ac = history.history['accuracy']
get_los = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(get_ac))
plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, get_ac, 'g', label='Accuracy of Training Data')
plt.plot(epochs, val_acc, 'r', label='Accuracy of Validation Data')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, get_los, 'g', label='Loss of Training Data')
plt.plot(epochs, val_loss, 'r', label='Loss of Validation Data')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()
plt.show()
```

Evaluate the Model:
```bash
loss, accuracy = model.evaluate(val_ds)
```

Display Predictions:
```bash
plt.figure(figsize=(20, 20))
for images, labels in val_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        score = tf.nn.softmax(predictions[0])
        if class_names[labels[i]] == class_names[np.argmax(score)]:
            plt.title("Actual: " + class_names[labels[i]])
            plt.ylabel("Predicted: " + class_names[np.argmax(score)], fontdict={'color': 'green'})
        else:
            plt.title("Actual: " + class_names[labels[i]])
            plt.ylabel("Predicted: " + class_names[np.argmax(score)], fontdict={'color': 'red'})
        plt.gca().axes.yaxis.set_ticklabels([])        
        plt.gca().axes.xaxis.set_ticklabels([])
```

Using the Model for prediction on local images:
```bash
#testing
img_path = 'Faulty_solar_panel/test_images/test_1.png'

# Load the image with the target size
img_height = 244
img_width = 244
img = image.load_img(img_path, target_size=(img_height, img_width))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add an extra dimension to the image (required for model input)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image using VGG16's preprocess_input function
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Predict the class using the model
predictions = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class index to the class label
predicted_label = class_names[predicted_class[0]]

# Print the predicted label
print(f"Predicted label: {predicted_label}")

# Plot the image with the predicted label
plt.imshow(image.array_to_img(img_array[0]))
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
```

Using the Model for prediction on Web-sourced images:
```bash
# Define the URL of the image
img_url ='https://www.yoururl.com/faulty_image.jpg'

# Download the image
response = requests.get(img_url)
# Load the image with the target size
img_height = 244
img_width = 244
img = image.load_img(BytesIO(response.content), target_size=(img_height, img_width))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add an extra dimension to the image (required for model input)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image using VGG16's preprocess_input function
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Predict the class using the model
predictions = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class index to the class label
predicted_label = class_names[predicted_class[0]]

# Print the predicted label
print(f"Predicted label: {predicted_label}")

# Plot the image with the predicted label
plt.imshow(image.array_to_img(img_array[0]))
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
```

## Conclusion
This project provides a comprehensive approach to classify the state of solar panels using a VGG16-based model. The model is trained and fine-tuned on a dataset of labeled images, achieving a satisfactory level of accuracy. The project includes data preparation, model definition, training, fine-tuning, evaluation, visualization of results and using the model for prediction.



