import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Constants
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
NUM_EPOCHS = 20
NUM_CLASS = 2
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'cnn_model.h5'

# Load CSV file
df = pd.read_csv('data/image_data.csv')

# Get image paths and corresponding labels
image_paths = ['data/cleaned_images/' + name for name in df['name']]
labels = df['category']

# Initialize empty lists to store processed images and labels
processed_images = []
processed_labels = []

# Process image data
for image_path, label in zip(image_paths, labels):
    image = load_img(image_path, target_size=(HEIGHT, WIDTH))
    image_array = img_to_array(image)
    image_array /= 255.0
    processed_images.append(image_array)
    processed_labels.append(label)

# Convert lists to NumPy arrays
processed_images = np.array(processed_images)
processed_labels = np.array(processed_labels)

# Load MobileNetV2 model (excluding top layers for transfer learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))

# Build a new model on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASS, activation='softmax')(x)

modelo = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with L2 regularization
modelo.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])

# Split data into training and test sets
datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    processed_images, processed_labels, test_size=0.2, random_state=42)

# Convert labels to binary format
etiquetas_entrenamiento = np.where(etiquetas_entrenamiento == 'aceptable', 0, 1)  # Swap 0 and 1
etiquetas_prueba = np.where(etiquetas_prueba == 'aceptable', 0, 1)  # Swap 0 and 1

# Train the model
modelo.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model
loss, accuracy = modelo.evaluate(datos_prueba, etiquetas_prueba)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Calculate additional metrics
y_pred = modelo.predict(datos_prueba)
y_pred_binary = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(etiquetas_prueba, y_pred_binary)
precision = precision_score(etiquetas_prueba, y_pred_binary)
recall = recall_score(etiquetas_prueba, y_pred_binary)
f1 = f1_score(etiquetas_prueba, y_pred_binary)
conf_matrix = confusion_matrix(etiquetas_prueba, y_pred_binary)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('Confusion Matrix:\n', conf_matrix)

# Save the model
modelo.save(MODEL_SAVE_PATH)
