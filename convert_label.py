import os

def convert_labels_in_place(input_dir, class_mapping):
    """
    Convertit les fichiers de labels en format YOLOv5 directement dans les fichiers originaux.

    Args:
        input_dir (str): Répertoire contenant les fichiers de labels à convertir.
        class_mapping (dict): Dictionnaire {nom_classe: index}.
    """
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, file_name)
        temp_path = input_path + ".temp"  # Fichier temporaire pour éviter d'écraser immédiatement

        with open(input_path, "r") as infile, open(temp_path, "w") as tempfile:
            for line in infile:
                parts = line.split()
                # Vérifier que la ligne a au moins 10 éléments
                if len(parts) < 10:
                    print(f"Skipping invalid line in {file_name}: {line.strip()}")
                    continue

                try:
                    # Extraire les sommets du polygone
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[2]), float(parts[3])
                    x3, y3 = float(parts[4]), float(parts[5])
                    x4, y4 = float(parts[6]), float(parts[7])

                    # Calculer les coordonnées de la bounding box
                    x_min = min(x1, x2, x3, x4)
                    y_min = min(y1, y2, y3, y4)
                    x_max = max(x1, x2, x3, x4)
                    y_max = max(y1, y2, y3, y4)

                    # Calculer x_center, y_center, width, height normalisés
                    img_width, img_height = 640, 640  # Assurez-vous que cette taille est correcte
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Convertir le nom de classe en indice
                    class_name = parts[8]
                    if class_name not in class_mapping:
                        print(f"Classe inconnue dans {file_name}: {class_name}")
                        continue

                    class_id = class_mapping[class_name]

                    # Écrire la ligne convertie dans le fichier temporaire
                    tempfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except ValueError as e:
                    print(f"Erreur de conversion dans {file_name}: {line.strip()} | Erreur : {e}")
                    continue

        # Remplacer le fichier original par le fichier temporaire
        os.replace(temp_path, input_path)
        print(f"Converti : {input_path}")


# Mapping des noms de classes aux indices
class_mapping = {
    "component": 0,
    "void": 1
}

# Répertoire des fichiers de labels
input_dir = "labelid_image/valid/labels"

# Conversion des labels directement dans les fichiers originaux
convert_labels_in_place(input_dir, class_mapping)
