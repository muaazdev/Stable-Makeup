from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import requests
import torch
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_plus import detail_encoder
from spiga_draw import spiga_process, spiga_segmentation
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector

app = FastAPI()

# External GenAI API for makeup analysis
GENAI_API_URL = "http://localhost:8001/analyze-makeup/"

# Initialize SPIGA and Face Detector once
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")
processor = SPIGAFramework(ModelConfig("300wpublic"))

# Model configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
MAKEUP_ENCODER_PATH = "./models/stablemakeup/pytorch_model.bin"
ID_ENCODER_PATH = "./models/stablemakeup/pytorch_model_1.bin"
POSE_ENCODER_PATH = "./models/stablemakeup/pytorch_model_2.bin"
IMAGE_ENCODER_DIR = "./models/image_encoder_l"
DEVICE = "cuda"

def get_draw(pil_img: Image.Image, size: int) -> Image.Image:
    spigas = spiga_process(pil_img, detector)
    if not spigas:
        w, h = pil_img.size
        return Image.new('RGB', (w, h), color=(0, 0, 0))
    return spiga_segmentation(spigas, size=size)

def infer_once(id_img: Image.Image, makeup_img: Image.Image) -> Image.Image:
    id_resized = id_img.resize((512, 512))
    makeup_resized = makeup_img.resize((512, 512))
    pose_img = get_draw(id_resized, size=512)
    return makeup_encoder.generate(
        id_image=[id_resized, pose_img],
        makeup_image=makeup_resized,
        pipe=pipe,
        guidance_scale=1.6
    )

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Load and prepare models on startup
Unet = OriginalUNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, IMAGE_ENCODER_DIR, DEVICE, dtype=torch.float32)

# Load custom weights
id_encoder.load_state_dict(torch.load(ID_ENCODER_PATH), strict=False)
pose_encoder.load_state_dict(torch.load(POSE_ENCODER_PATH), strict=False)
makeup_encoder.load_state_dict(torch.load(MAKEUP_ENCODER_PATH), strict=False)

# Move to device
id_encoder.to(DEVICE)
pose_encoder.to(DEVICE)
makeup_encoder.to(DEVICE)

# Setup Stable Diffusion ControlNet pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_ID,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32
).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

@app.post("/transfer-makeup/")
async def transfer_makeup(
    no_makeup_img: UploadFile = File(...),
    makeup_img: UploadFile = File(...)
):
    # Read and decode images
    no_makeup_bytes = await no_makeup_img.read()
    makeup_bytes = await makeup_img.read()
    no_makeup_pil = Image.open(io.BytesIO(no_makeup_bytes)).convert("RGB")
    makeup_pil = Image.open(io.BytesIO(makeup_bytes)).convert("RGB")

    # Keep original size for output
    original_size = no_makeup_pil.size  # (width, height)

    # Run inference and resize back
    output_img = infer_once(no_makeup_pil, makeup_pil)
    output_img = output_img.resize(original_size, resample=Image.BICUBIC)

    # Encode to base64
    output_base64 = pil_to_base64(output_img)

    # Call external GenAI makeup analysis
    genai_steps = None
    try:
        files = {'image': ('makeup.jpg', makeup_bytes, makeup_img.content_type)}
        response = requests.post(GENAI_API_URL, files=files, timeout=30)
        if response.status_code == 200:
            genai_steps = response.json()
        else:
            genai_steps = f"GenAI API failed: {response.status_code}"
    except Exception as e:
        genai_steps = f"Error calling GenAI API: {str(e)}"

    # Return image and steps
    return JSONResponse(content={
        "image": output_base64,
        "steps": genai_steps
    })
