from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from werkzeug.utils import secure_filename
import warnings

# Initialisation de Flask
app = Flask(
    __name__,
    template_folder='templates',  # Chemin des fichiers HTML
    static_folder='static'       # Chemin des fichiers statiques
)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle SAM
MODEL_TYPE = "vit_b"
MODEL_PATH = os.path.join('models', 'sam_vit_b_01ec64.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Chargement du modèle SAM...")
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
except TypeError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")

# Initialiser et charger le modèle
sam = sam_model_registry[MODEL_TYPE]()
sam.load_state_dict(state_dict, strict=False)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Modèle SAM chargé avec succès!")

# Fonction pour générer une couleur unique pour chaque classe
def get_color_for_class(class_name):
    np.random.seed(hash(class_name) % (2**32))
    return tuple(np.random.randint(0, 256, size=3).tolist())

# Convertir un masque en bounding box au format YOLOv5
def mask_to_yolo_bbox(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # YOLOv5 format: x_center, y_center, width, height (normalized)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            return "Aucun fichier sélectionné", 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', uploaded_image=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    image_name = data.get('image_name')
    points = data.get('points')

    if not image_name or not points:
        return jsonify({'success': False, 'error': 'Données manquantes'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image non trouvée'}), 404

    # Créer un dossier pour sauvegarder les résultats
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(image_name)[0])
    os.makedirs(output_dir, exist_ok=True)

    # Charger l'image et effectuer la segmentation
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    annotated_image = image.copy()

    # YOLOv5 annotation
    yolo_annotations = []

    for point in points:
        x, y = point['x'], point['y']
        class_name = point.get('class', 'Unknown')
        class_id = hash(class_name) % 1000  # Générer un ID unique basé sur le nom
        color = get_color_for_class(class_name)  # Couleur unique pour chaque classe
        masks, _, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        mask = masks[0]
        annotated_image[mask > 0] = color  # Superposer le masque avec la couleur

        # Convertir le masque en bounding box YOLOv5
        bbox = mask_to_yolo_bbox(mask)
        if bbox:
            x_center, y_center, width, height = bbox
            # Normaliser les valeurs
            x_center /= image.shape[1]
            y_center /= image.shape[0]
            width /= image.shape[1]
            height /= image.shape[0]
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Ajouter le texte de la classe
        cv2.putText(annotated_image, class_name, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Texte blanc

    # Sauvegarder les résultats
        annotated_filename = f"annotated_{image_name}"
        annotated_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)

        # Sauvegarder les annotations YOLOv5
        yolo_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(yolo_path, "w") as f:
            f.write("\n".join(yolo_annotations))

        # Copier l'image originale dans le dossier
        original_copy_path = os.path.join(output_dir, image_name)
        if not os.path.exists(original_copy_path):
            os.rename(image_path, original_copy_path)

        # Renvoyer le chemin relatif pour affichage
        relative_output_dir = output_dir.replace("static/", "")
        return jsonify({
            'success': True,
            'output_dir': relative_output_dir,
            'annotated_image': f"{relative_output_dir}/{annotated_filename}"
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
