
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

    prompt = "masterpiece, best_quality"
    if options.get('gender') is not None: 
        prompt += ", " + options['gender']
    if options.get('hair') is not None: 
        prompt += ", " + options['hair']
    if options.get('accessories') and len(options['accessories']) != 0: 
        prompt += ", " + ", ".join(options['accessories'])

    list_prompt = []
    if options.get('emotion') is not None and len(options['emotion']) != 0:
        for each in options['emotion']:
            new_prompt = prompt + ", " each
            list_prompt.append(new_prompt)
            
    else : list_prompt.append(prompt)

    negative_prompt = 'nsfw, bad anatomy, worst quality'
    strength = options['strength']

    # Open the uploaded image
    image_data = await file.read()
    init_image = Image.open(io.BytesIO(image_data))

    res = []
    for prompt in list_prompt:
        # Generate Image
        step2_image = model.img2img(image=init_image, prompt=prompt, negative_prompt=negative_prompt, strength=strength)
        # Convert generated image to bytes
        img_byte_arr = io.BytesIO()
        step2_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        res.append(StreamingResponse(img_byte_arr, media_type="image/png"))
        
    print("Processing Real to Anime Image Successfully")
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
