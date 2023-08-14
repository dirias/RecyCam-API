import os
import csv

def create_csv ():

    input_folder = 'data/cleaned_images'
    output_csv = 'data/image_data.csv'

    # Verificar si el archivo CSV ya existe, si existe, eliminarlo
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Abrir el archivo CSV en modo escritura
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'category'])  # Escribir los encabezados en la primera fila del CSV

        # Recorrer todos los archivos en la carpeta de entrada
        for filename in os.listdir(input_folder):
            if filename.endswith('.jpg'):
                name = filename
                _, category = filename.split('_')[:2]  # Extraer el nombre y la categoría del archivo
                category = category.split('.')[0]  # Eliminar la extensión del archivo

                # Escribir el nombre y la categoría en una nueva fila del CSV
                writer.writerow([name, category])

    print(f"Se ha creado el archivo CSV '{output_csv}' con los datos de las imágenes.")
