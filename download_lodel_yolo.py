from huggingface_hub import hf_hub_download
import os

def download_yolo_model(repo_id="keizer77/samyolo2", filename="best.pt", output_dir="models"):
    """
    Télécharge un modèle YOLOv5 depuis Hugging Face.
    """
    os.makedirs(output_dir, exist_ok=True)  # Créez le dossier si nécessaire

    print(f"Téléchargement du modèle {filename} depuis le dépôt {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=output_dir)
    print(f"Modèle téléchargé et sauvegardé dans : {model_path}")
    return model_path

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        downloaded_model_path = download_yolo_model()
        print(f"Modèle prêt à être utilisé : {downloaded_model_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle : {e}")
