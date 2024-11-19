from flask import Flask, request, render_template, jsonify, send_from_directory,url_for
import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from werkzeug.utils import secure_filename
import warnings

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

# Générer une couleur unique pour chaque classe
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
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

@app.route('/', methods=['GET', 'POST'])
def index():
    """Page principale pour télécharger et afficher les images."""
    if request.method == 'POST':
        files = request.files.getlist('images')
        if not files:
            return "Aucun fichier sélectionné", 400

        filenames = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filename)

        return render_template('index.html', uploaded_images=filenames)

    uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', uploaded_images=uploaded_images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir les fichiers uploadés."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/segment', methods=['POST'])
def segment():
    """Endpoint pour effectuer la segmentation des images."""
    try:
        data = request.get_json()
        print("Données reçues :", data)

        if not isinstance(data, list):
            return jsonify({'success': False, 'error': 'Format incorrect : liste attendue'}), 400

        output = []

        for item in data:
            image_name = item.get('image_name')
            points = item.get('points', [])

            if not image_name or not points:
                return jsonify({'success': False, 'error': f"Données manquantes pour l'image {image_name}"}), 400

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
            if not os.path.exists(image_path):
                return jsonify({'success': False, 'error': f"Image {image_name} non trouvée"}), 404

            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                return jsonify({'success': False, 'error': f"Impossible de charger l'image {image_name}"}), 400

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            annotated_image = image.copy()
            yolo_annotations = []

            for point in points:
                x, y = point['x'], point['y']
                class_name = point.get('class', 'Unknown')
                color = get_color_for_class(class_name)

                try:
                    masks, _, _ = predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    mask = masks[0]
                    annotated_image[mask > 0] = color

                    # Convertir le masque en bounding box YOLOv5
                    bbox = mask_to_yolo_bbox(mask)
                    if bbox:
                        x_center, y_center, width, height = bbox
                        x_center /= image.shape[1]
                        y_center /= image.shape[0]
                        width /= image.shape[1]
                        height /= image.shape[0]
                        yolo_annotations.append(f"{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                except Exception as e:
                    print(f"Erreur de segmentation pour le point {point} : {e}")

            # Sauvegarder les résultats
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(image_name)[0])
            os.makedirs(output_dir, exist_ok=True)
            annotated_path = os.path.join(output_dir, f"annotated_{image_name}")
            cv2.imwrite(annotated_path, annotated_image)

            yolo_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
            with open(yolo_path, "w") as f:
                f.write("\n".join(yolo_annotations))
            
            new_image_path = os.path.join(output_dir, image_name)
            if not os.path.exists(new_image_path):
                os.rename(image_path, new_image_path)

            output.append({
                'image_name': image_name,
                'annotated_image': url_for('static', filename=f"uploads/{os.path.splitext(image_name)[0]}/annotated_{image_name}"),
                'yolo_annotations': url_for('static', filename=f"uploads/{os.path.splitext(image_name)[0]}/{os.path.splitext(image_name)[0]}.txt")
            })

        return jsonify({'success': True, 'results': output})

    except Exception as e:
        print("Erreur dans /segment :", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)