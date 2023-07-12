import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Constantes
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
NUM_EPOCHS = 5
NUM_CLASS = 2
tamaño_lote = 32

# Cargar el archivo CSV
df = pd.read_csv('../data/image_data.csv')

# Obtener las rutas de las imágenes y las etiquetas correspondientes
image_paths = ['../data/cleaned_images/' + name for name in df['name']]
labels = df['category']

# Crear listas vacías para almacenar las imágenes y las etiquetas preprocesadas
processed_images = []
processed_labels = []

# Recorrer las rutas de las imágenes
for image_path, label in zip(image_paths, labels):
    # Cargar la imagen
    image = load_img(image_path, target_size=(HEIGHT, WIDTH))
    # Convertir la imagen a un arreglo NumPy
    image_array = img_to_array(image)
    # Normalizar los valores de los píxeles entre 0 y 1
    image_array /= 255.0
    # Agregar la imagen preprocesada y la etiqueta a las listas correspondientes
    processed_images.append(image_array)
    processed_labels.append(label)

# Convertir las listas a arreglos NumPy
processed_images = np.array(processed_images)
processed_labels = np.array(processed_labels)

import pdb; pdb.set_trace()

# Definir el modelo CNN
def crear_modelo_cnn():
    """
    altura: Representa la altura de las imágenes que se utilizarán como entrada para el modelo CNN. 
    Debes definir este valor según las dimensiones de tus imágenes.

    anchura: Representa la anchura de las imágenes que se utilizarán como entrada para el modelo CNN. 
    Debes definir este valor según las dimensiones de tus imágenes.

    canales: Representa el número de canales de las imágenes que se utilizarán como entrada para el modelo CNN. 
    Si las imágenes son en escala de grises, el valor típico es 1. Si las imágenes son en color (RGB), el valor típico es 3 (uno para cada canal de rojo, verde y azul).

    num_clases: Representa el número de clases o categorías diferentes a las que deseas clasificar tus imágenes. 
    Debes definir este valor según la cantidad de clases que tienes en tu conjunto de datos.
    """
    modelo = tf.keras.Sequential()
    
    # Capa convolucional 1
    modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, CHANNELS)))
    modelo.add(layers.MaxPooling2D((2, 2)))
    
    # Capa convolucional 2
    modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
    modelo.add(layers.MaxPooling2D((2, 2)))
    
    # Capa convolucional 3
    modelo.add(layers.Conv2D(128, (3, 3), activation='relu'))
    modelo.add(layers.MaxPooling2D((2, 2)))
    
    # Capa completamente conectada
    modelo.add(layers.Flatten())
    modelo.add(layers.Dense(128, activation='relu'))
    modelo.add(layers.Dense(NUM_CLASS, activation='softmax'))
    
    return modelo

# Crear el modelo
modelo = crear_modelo_cnn()

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# Dividir los datos en conjuntos de entrenamiento y prueba
datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    processed_images, processed_labels, test_size=0.2, random_state=42)

import pdb; pdb.set_trace()

# Entrenar el modelo
modelo.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=NUM_EPOCHS, batch_size=tamaño_lote)

# Evaluar el modelo
pérdida, precisión = modelo.evaluate(datos_prueba, etiquetas_prueba)
print('Pérdida:', pérdida)
print('Precisión:', precisión)

# Utilizar el modelo para hacer predicciones
predicciones = modelo.predict(datos_nuevas_imagenes)
etiqueta_predicha = etiquetas_clases[np.argmax(predicciones)]
print('Etiqueta predicha:', etiqueta_predicha)