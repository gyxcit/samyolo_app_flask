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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Aucun fichier sélectionné", 400
        file = request.files['image']
        if file.filename == '':
            return "Nom de fichier vide", 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Passer le nom du fichier au template pour affichage
        return render_template('index.html', uploaded_image=filename)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/segment', methods=['POST'])
def segment():
    """Endpoint pour segmenter une image et sauvegarder les annotations."""
    try:
        data = request.get_json()
        image_name = data.get('image_name')
        points = data.get('points')

        if not image_name or not points:
            return jsonify({'success': False, 'error': 'Données manquantes'}), 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image non trouvée'}), 404

        # Charger l'image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Annoter l'image avec les masques et les classes
        annotated_image = image.copy()
        for point in points:
            x, y = point['x'], point['y']
            class_name = point.get('class', 'Unknown')
            input_points = np.array([[x, y]])
            input_labels = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
            mask = masks[0]
            mask_image = (mask * 255).astype(np.uint8)

            # Superposer le masque à l'image
            color = (0, 255, 0)  # Couleur verte pour les masques
            annotated_image[mask > 0] = color

            # Ajouter le texte de la classe
            cv2.putText(annotated_image, class_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Sauvegarder l'image annotée
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{image_name}")
        cv2.imwrite(annotated_path, annotated_image)

        return jsonify({'success': True, 'annotated_image': f"annotated_{image_name}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
