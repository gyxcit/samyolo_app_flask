<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Labélisation d'Images avec SAM</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }

        h1 {
            text-align: center;
            color: #333;
        }

        section {
            margin: 20px 0;
            padding: 20px;
            background: #fff;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }

        canvas {
            border: 2px solid #ddd;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
        }

        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #45a049;
        }

        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }

        .upload-section, .class-management, .controls {
            text-align: center;
        }

        .class-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            list-style: none;
            padding: 0;
        }

        .class-item {
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #f4f4f4;
            border: 1px solid #ccc;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .class-item:hover {
            background-color: #ddd;
        }

        .class-item.active {
            background-color: #4CAF50;
            color: white;
            border-color: #45a049;
        }

        #annotated-image {
            display: block;
            max-width: 100%;
            margin: 20px auto;
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }

        .status {
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }

        .status.success {
            color: #4CAF50;
        }

        .status.error {
            color: #f44336;
        }

        .image-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }

        .image-item {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        /* Limite la taille des images téléchargées */
        .image-item img {
            max-width: 100%; /* S'adapte à la taille du conteneur */
            max-height: 100%; /* Empêche de dépasser le conteneur */
            border-radius: 8px;
            object-fit: cover; /* Maintient le rapport d'aspect */
        }

        /* Taille fixe des conteneurs pour les images */
        .image-item {
            width: 150px;
            height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden; /* Coupe les parties excédentaires */
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .image-item:hover {
            border-color: #4CAF50;
        }

        /* Ajout pour le conteneur général des images téléchargées */
        .image-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Labélisation d'Images avec SAM</h1>

    <!-- Section 1: Upload Images -->
    <section class="upload-section">
        <form method="post" enctype="multipart/form-data">
            <label for="image">Télécharger des images :</label><br>
            <input type="file" id="image" name="images" accept="image/*" multiple required>
            <br>
            <button type="submit">Télécharger les images</button>
        </form>
    </section>

    {% if uploaded_images %}
    <!-- Section 2: Images déjà téléchargées -->
    <section>
        <h2>Images téléchargées</h2>
        <div class="image-container">
            {% for image in uploaded_images %}
                <div class="image-item" onclick="loadImage('{{ image }}')">
                    <img src="{{ url_for('static', filename='uploads/' + image) }}" alt="{{ image }}">
                </div>
            {% endfor %}
        </div>
    </section>

    <!-- Section 3: Canvas for Image Annotation -->
    <section>
        <canvas id="image-canvas"></canvas>
    </section>

    <!-- Section 4: Annotated Image -->
    <section>
        <h2>Image Annotée</h2>
        <img id="annotated-image" style="display: none;" alt="Image Annotée">
    </section>

    <!-- Section 5: Manage Classes -->
    <section class="class-management">
        <h3>Ajouter une classe :</h3>
        <input type="text" id="class-name" placeholder="Entrez une classe">
        <button id="add-class">Ajouter</button>
        <h4>Classes disponibles :</h4>
        <ul id="class-list" class="class-list"></ul>
    </section>

    <!-- Section 6: Controls -->
    <section>
        <button id="finish-button" disabled>Terminer l'annotation</button>
        <button id="segment-button" disabled>Lancer la Segmentation</button>
    </section>
    {% endif %}

    <script>
        let canvas = document.getElementById('image-canvas');
        let ctx = canvas ? canvas.getContext('2d') : null;
        let points = [];
        let currentClass = null;
        let img = null;
        let imgWidth = 0;
        let imgHeight = 0;

        {% if uploaded_images %}
        function loadImage(imageName) {
            // Réinitialiser le tableau des points
            points = [];
            document.getElementById('finish-button').disabled = false;
            document.getElementById('segment-button').disabled = true;

            // Charger l'image sélectionnée
            img = new Image();
            img.src = "{{ url_for('static', filename='uploads/') }}" + imageName;

            img.onload = () => {
                imgWidth = img.width;
                imgHeight = img.height;
                canvas.width = imgWidth;
                canvas.height = imgHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            };
        }

        // Gestion des classes
        document.getElementById('add-class').addEventListener('click', () => {
            const className = document.getElementById('class-name').value.trim();
            if (!className) return;

            const li = document.createElement('li');
            li.textContent = className;
            li.classList.add('class-item');
            li.onclick = () => {
                document.querySelectorAll('.class-item').forEach(item => item.classList.remove('active'));
                li.classList.add('active');
                currentClass = className;
            };
            document.getElementById('class-list').appendChild(li);

            document.getElementById('class-name').value = '';
        });

        // Ajouter un point sur le canvas
        canvas.addEventListener('click', event => {
            if (!currentClass) {
                alert('Veuillez sélectionner une classe.');
                return;
            }

            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            points.push({ x, y, class: currentClass });

            ctx.fillStyle = 'red';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();

            // Activer le bouton Segmentation
            document.getElementById('segment-button').disabled = points.length === 0;
        });

        // Terminer l'annotation
        document.getElementById('finish-button').addEventListener('click', () => {
            alert("Annotation terminée pour cette image.");
            document.getElementById('finish-button').disabled = true;
        });

        // Lancer la segmentation
        document.getElementById('segment-button').addEventListener('click', () => {
            fetch('/segment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_names: ["{{ uploaded_image }}"],  // Passer l'image actuelle
                    points: points
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('annotated-image').src = "{{ url_for('static', filename='') }}" + data.annotated_image;
                    document.getElementById('annotated-image').style.display = 'block';
                } else {
                    alert("Erreur : " + data.error);
                }
            })
            .catch(err => {
                console.error('Erreur:', err);
            });
        });
        {% endif %}
    </script>
</body>
</html>
