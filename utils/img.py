from PIL import Image # This requires Pillow library
import os

# Ejemplo de uso
TRASH_TYPE = 'plastic'
TRASH_CATEGORY = 'aceptable'
INPUT_FOLDER = f'../data/uncleaned_images/{TRASH_TYPE}/{TRASH_CATEGORY}'
OUTPUT_FOLDER = f'../data/cleaned_images'
NEW_WIDTH = 512
NEW_HEIGHT = 512

def resize_images(input_folder, output_folder, new_width, new_height):
    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Inicializar contador para nombres de imágenes
    image_counter = 1

    # Recorrer todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        # Verificar si el archivo es una imagen .jpg
        if filename.endswith('.jpg'):
            # Abrir la imagen
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            # Redimensionar la imagen
            resized_img = img.resize((new_width, new_height))

            # Generar el nuevo nombre de archivo
            new_filename = f"{TRASH_TYPE}_{TRASH_CATEGORY}_{image_counter}.jpg"
            image_counter += 1

            # Guardar la imagen redimensionada con el nuevo nombre en la carpeta de salida
            output_path = os.path.join(output_folder, new_filename)
            resized_img.save(output_path)

            print(f"Imagen {filename} redimensionada y guardada como {new_filename} en {output_path}")

resize_images(INPUT_FOLDER, OUTPUT_FOLDER, NEW_WIDTH, NEW_HEIGHT)
