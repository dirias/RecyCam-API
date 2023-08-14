# RecyCam-API
Implementation of a Convolutional Neural Network for image recognition to classify
if an object (trash) is ready or not to be recycled.

# Server

A Fast-API server was built to server the Convolutional Neural Network.

## Endpoints

### /predict/

Based on a picture, the model clasifies if the object is acceptable to be
recycled.

Expected return:
    {
        "predictions": 'Prediction percentage',
        "predicted_class": 'Prediction, either Acceptable or Not Acceptable'
    }



# Contribuiters

- Didier Irias Méndez
- Jeffrey Baltodano
