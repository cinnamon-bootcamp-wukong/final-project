from fastapi import FastAPI, UploadFile, Form
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from src.face_dect import face_model
import json
import yaml
import io
import base64

# Load configuration from YAML file
with open("src/training_configs.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.SafeLoader)

# Initialize FastAPI app
app = FastAPI()
face_model = face_model.FaceDetector(
    prototxt_path="src/face_dect/deploy.prototxt",
    model_path="src/face_dect/res10_300x300_ssd_iter_140000.caffemodel",
)

# Configure CORS
origins = [
    "http://0.0.0.0",
    "http://0.0.0.0:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def home():
    return {"message": "Hello World"}


@app.post("/face_dect/")
async def faceDect(file: UploadFile):
    image_data = await file.read()
    np_img = np.frombuffer(image_data, np.uint8)
    value = face_model.detect_faces(np_img)
    return {"is_face": bool(value)}


@app.post("/real2anime/")
async def real2anime(file: UploadFile, option_json: str = Form(...)):
    # Load the options from the provided JSON string
    options = json.loads(option_json)

    # Build the prompt based on options
    prompt = ""

    if options.get('gender') is not None:
        if options['gender'] == "male":
            prompt += "1boy"
        else:
            prompt += "1girl"

    if options.get('hair') is not None:
        prompt += ", " + options['hair'] + " hair"

    prompt += ", looking at viewer"

    list_prompt = []
    if options.get('emote') is not None and len(options['emote']) != 0:
        for each in options['emote']:
            new_prompt = prompt + ", " + each
            list_prompt.append(new_prompt)

    else:
        list_prompt.append(prompt)

    for idx in range(len(list_prompt)):
        list_prompt[idx] += ", upper body"

        if options.get('accessories') and len(options['accessories']) != 0:
            list_prompt[idx] += ", " + ", ".join(options['accessories'])

        list_prompt[idx] += ", masterpiece"

        if options.get('age') is not None:
            list_prompt[idx] += ", " + options['age'] + " year olds"

    negative_prompt = 'nsfw'
    strength = options['strength']

    # Open the uploaded image
    image_data = await file.read()
    init_image = Image.open(io.BytesIO(image_data))

    images_base64 = []

    for idx, prompt in enumerate(list_prompt):
        step2_image = Image.open("arya.png")  # Replace with your actual image generation logic

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        step2_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Encode image to base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        images_base64.append(img_base64)

    return {"images": images_base64}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
