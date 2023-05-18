import tensorflow as tf
from tensorflow.keras import layers

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
    modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(altura, anchura, canales)))
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
    modelo.add(layers.Dense(num_clases, activation='softmax'))
    
    return modelo

# Crear el modelo
modelo = crear_modelo_cnn()

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=num_epochs, batch_size=tamaño_lote)

# Evaluar el modelo
pérdida, precisión = modelo.evaluate(datos_prueba, etiquetas_prueba)
print('Pérdida:', pérdida)
print('Precisión:', precisión)

# Utilizar el modelo para hacer predicciones
predicciones = modelo.predict(datos_nuevas_imagenes)
etiqueta_predicha = etiquetas_clases[np.argmax(predicciones)]
print('Etiqueta predicha:', etiqueta_predicha)