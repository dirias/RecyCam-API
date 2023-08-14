# RecyCam-API
Implementation of a Convolutional Neural Network for image recognition to classify
if an object (trash) is ready or not to be recycled.

# Server

A Fast-API server was built to serve the Convolutional Neural Network.

## Runtime Instructions

### 1. Install dependencies

pip install -r requirements.txt

### 2. Download the dataset

**dataset link**: https://drive.google.com/drive/folders/1KN3J5OmE2HQUOh1K175mjU2xZ5WoyAk8?usp=sharing

- Place the dataset inside the _**data**_ folder.
- Run <pending>

### 3. Run the server

- Move inside the _**app**_ folder.
- Start the server with <code>uvicorn app:app --host [COMPUTER_IP] --port [DISIRE_PORT]</code>
- Example: <code>uvicorn app:app --host 192.168.0.6 --port 8000</code>

## Endpoints

### /predict/

Based on a picture, the model classifies if the object is acceptable or not to be
recycled.

Expected return:
    {
        "predictions": 'Prediction percentage',
        "predicted_class": 'Prediction, either Acceptable or Not Acceptable'
    }



# Contribuiters

- Didier Irias Méndez
- Jeffrey Baltodano
