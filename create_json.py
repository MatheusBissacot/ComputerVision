import os
import json

def get_image_paths(folder_path, output_json):
    # Lista para armazenar os caminhos das imagens
    image_paths = {"images": []}
    
    # Percorre todos os arquivos na pasta
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths["images"].append({"path": os.path.join(root, file)})
    
    # Salva os caminhos no arquivo JSON
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(image_paths, json_file, indent=4)
    
    print(f"Arquivo JSON salvo: {output_json}")

# Defina o caminho da pasta contendo as imagens e o nome do JSON de saída
folder_path = "VC_2425_Project_public"
output_json = "image_paths.json"

get_image_paths(folder_path, output_json)
