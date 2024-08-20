
from fastapi import FastAPI, UploadFile, Form
from src.model.wrapper import SDXLModel
from fastapi.responses import StreamingResponse
from PIL import Image
import json
import yaml
import torch
import io

# Set a manual seed for reproducibility
torch.manual_seed(42)

# Load configuration from YAML file
with open("src/training_configs.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.SafeLoader)

# Initialize FastAPI app and model
app = FastAPI()
model = SDXLModel(**cfg["model"])

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.post("/real2anime)
async def real2anime(file: UploadFile, option_json: str = Form(...)):
    # Load the options from the provided JSON string
    options = json.loads(option_json)

    prompt = "best_quality"
    if options.get('age') is not None: 
        prompt += ", " + options['age']
    if options.get('gender') is not None: 
        prompt += ", " + options['gender']
    if options.get('accessories') and len(options['accessories']) != 0: 
        prompt += ", " + ", ".join(options['accessories'])

    negative_prompt = 'nsfw'
    strength = 0.75

    # Open the uploaded image
    image_data = await file.read()
    init_image = Image.open(io.BytesIO(image_data))

    step2_image = model.img2img(image=init_image, prompt=prompt, negative_prompt=negative_prompt, strength=strength)
    print("Processing Real to Anime Image Successfully")

    # Convert generated image to bytes
    img_byte_arr = io.BytesIO()
    step2_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/anime2emotion")
async def anime2emotion(file: UploadFile, selected_emotions: str = Form(...)):
    selected_emotions = json.loads(option_json) # example: value : ['sad']
    if len(selected_emotions['value']) == 0 : selected_emotions['value'] = ['happy'] # Default value is happy
        
    prompt = "best_quality"
    prompt += ", " + ", ".join(selected_emotions['value'])
    
    content = await file.read()
    init_image = Image.open(io.BytesIO(content))

    negative_prompt = 'nsfw'
    strength = 0.75

    generated_image = model.img2img(image=init_image, prompt=prompt, negative_prompt=negative_prompt, strength=strength)
    print("Adding Emotion to Anime Image Successfully")

    # Convert generated image to bytes
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
