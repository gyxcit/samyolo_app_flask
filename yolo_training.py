import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import sys
import os

# Ajoutez le chemin du répertoire YOLOv5 au sys.path pour pouvoir importer train.py
BASE_DIR = Path(__file__).resolve().parent
YOLOV5_DIR = BASE_DIR / "yolov5"
sys.path.append(str(YOLOV5_DIR))

from yolov5.train import main, parse_opt  # Importation directe des fonctions nécessaires

# Définir les chemins et options
MODEL_PATH = BASE_DIR / "models/models--keizer77--samyolo2/snapshots/74c8cb12ae448ff0b8bae9ef522b54ec09b47c20/best.pt"
DATA_YAML_PATH = BASE_DIR / "labelid_image/data.yaml"
OUTPUT_DIR = BASE_DIR / "weights"

def clear_cache(data_path):
    """
    Supprime les fichiers de cache de labels pour s'assurer que YOLOv5
    recrée les caches à partir des fichiers d'annotation actuels.
    """
    subfolders = ['train', 'valid', 'test']
    for folder in subfolders:
        cache_file = os.path.join(data_path, folder, 'labels.cache')
        if os.path.exists(cache_file):
            print(f"Suppression du cache : {cache_file}")
            os.remove(cache_file)

def train_yolo_direct():
    # Nettoyer le cache avant l'entraînement
    clear_cache("labelid_image")

    # Préparer les options pour l'entraînement
    opt = parse_opt()
    opt.imgsz = 640
    opt.batch_size = 8
    opt.epochs = 10
    opt.data = str(DATA_YAML_PATH)
    opt.weights = str(MODEL_PATH)
    opt.project = str(OUTPUT_DIR)
    opt.name = "custom_model"
    opt.device = "cpu"  # Spécifier le périphérique (CPU ou GPU)

    print("Lancement de l'entraînement YOLOv5...")
    main(opt)

if __name__ == "__main__":
    try:
        train_yolo_direct()
    except Exception as e:
        print(f"Erreur lors de l'exécution de l'entraînement : {e}")
