from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app=FastAPI()
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "https://remaining-nedda-bakhtiyorjon-3f910bd5.koyeb.app"
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

MODEL=tf.keras.models.load_model("./cnn_model.keras")
CLASS_NAMES=['Ealy Blight',"Late Blight","Healthy"]

def read_file_as_image(data)-> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")  
async def predict(
    file: UploadFile=File(...)
):
    image= read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    predictions = MODEL.predict(image_batch)
    result_ind=int(np.argmax(predictions[0]))
    predicted_class=str(CLASS_NAMES[result_ind])
    confidence=float(np.max(predictions[0]))

    return {
        "class":predicted_class,
        "confidence":confidence
    }
# Run the server
if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
