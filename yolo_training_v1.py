import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import sys

# Ajoutez le chemin du répertoire YOLOv5 au sys.path pour pouvoir importer train.py
BASE_DIR = Path(__file__).resolve().parent
YOLOV5_DIR = BASE_DIR / "yolov5"
sys.path.append(str(YOLOV5_DIR))

from yolov5.train import main, parse_opt  # Importation directe des fonctions nécessaires

# Définir les chemins et options
MODEL_PATH = BASE_DIR / "models/models--keizer77--samyolo2/snapshots/74c8cb12ae448ff0b8bae9ef522b54ec09b47c20/best.pt"
DATA_YAML_PATH = BASE_DIR / "labelid_image/data.yaml"
OUTPUT_DIR = BASE_DIR / "weights"

def train_yolo_direct():
    # Préparer les options pour l'entraînement
    opt = parse_opt()
    opt.imgsz = 640
    opt.batch_size = 16
    opt.epochs = 10
    opt.data = str(DATA_YAML_PATH)
    opt.weights = str(MODEL_PATH)
    opt.project = str(OUTPUT_DIR)
    opt.name = "custom_model"
    opt.device = "cpu"  # Spécifier le périphérique (CPU ou GPU)
    
    # Appeler directement la fonction d'entraînement
    print("Lancement de l'entraînement YOLOv5...")
    main(opt)

if __name__ == "__main__":
    try:
        train_yolo_direct()
    except Exception as e:
        print(f"Erreur lors de l'exécution de l'entraînement : {e}")
