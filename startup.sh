#!/bin/bash

echo "Starting the deployment process"

# 1. Créer une virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# 2. Activer la virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Installer les dépendances
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Exécuter les fichiers Python contenant 'download' dans leur nom
echo "Running download scripts..."
for file in $(ls | grep download.*\.py); do
    echo "Executing $file..."
    python $file
done

# 5. Démarrer l'application avec Gunicorn
echo "Starting the Flask application with Gunicorn..."
gunicorn --bind 0.0.0.0:8000 app:app
