from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from logging import getLogger

log = getLogger(__name__)

PREDICTION_CATEGORIES = {0: 'No Aceptable',
                         1: 'Aceptable'}

app = FastAPI()

# Load the TensorFlow.js model
model = load_model('../cnn_model.h5')

# Configure CORS settings
origins = [
    "http://localhost",  # Replace with the origin of your React Native app
    "http://localhost:3000",  # Another example
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    image = image.resize((512, 512))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the image file and preprocessfil it
        image = Image.open(io.BytesIO(await file.read()))
        preprocessed_image = preprocess_image(image)
        #import pdb; pdb.set_trace()
        # Convert the NumPy array back to a PIL Image object
        preprocessed_pil_image = Image.fromarray(np.uint8(preprocessed_image[0] * 255))
        
        # Save the preprocessed image as a file for debugging
        preprocessed_pil_image.save("preprocessed_image.jpg")
        
        # Make predictions using the model
        predictions = model.predict(preprocessed_image)
        prediction_classes = np.argmax(predictions, axis=1)
        log.warning(f'Return {PREDICTION_CATEGORIES.get(prediction_classes.tolist()[0])} for the prediction.')
        #import pdb; pdb.set_trace()
        return {
            "predictions": np.round((predictions * 100), 2).tolist()[0],
            "predicted_classes": PREDICTION_CATEGORIES.get(prediction_classes.tolist()[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing image: " + str(e))
