from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = r"D:\Study\PBL_1_5th_semester_2023-24_Potato_Plant Disease_Dietection_project\Project\models\final1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3-channel image
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image, dtype=np.float32)  # Convert to NumPy array
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension
        img_batch = img_batch / 255.0  # Normalize pixel values

        prediction = MODEL.predict(img_batch)  # Get prediction
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]) * 100)  # Convert to percentage

        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
