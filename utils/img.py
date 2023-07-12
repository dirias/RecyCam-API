from PIL import Image # This requires Pillow library
import os

# Ejemplo de uso
INPUT_FOLDER = '../data/uncleaned_images'
OUPUT_FOLDER = '../data/cleaned_images'
NEW_WIDTH = 256
NEW_HEIGHT = 256

def resize_images(input_folder, output_folder, new_width, new_height):
    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        # Verificar si el archivo es una imagen .jpg
        if filename.endswith('.jpg'):
            # Abrir la imagen
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            # Redimensionar la imagen
            resized_img = img.resize((new_width, new_height))

            # Guardar la imagen redimensionada en la carpeta de salida
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

            print(f"Imagen {filename} redimensionada y guardada en {output_path}")

resize_images(INPUT_FOLDER, OUPUT_FOLDER, NEW_WIDTH, NEW_HEIGHT)
