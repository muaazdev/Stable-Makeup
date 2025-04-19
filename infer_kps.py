import os
import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_plus import detail_encoder
from spiga_draw import *
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector

# Setup SPIGA keypoint processor and face detector
processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")

# Helper functions
def get_draw(pil_img, size):
    spigas = spiga_process(pil_img, detector)
    if spigas is False:
        width, height = pil_img.size  
        return Image.new('RGB', (width, height), color=(0, 0, 0))  
    else:
        return spiga_segmentation(spigas, size=size)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def concatenate_images(image_files, output_file):
    images = image_files
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

# Use the official Hugging Face model ID
model_id = "runwayml/stable-diffusion-v1-5"

# Paths to custom model weights
makeup_encoder_path = "./models/stablemakeup/pytorch_model.bin"
id_encoder_path = "./models/stablemakeup/pytorch_model_1.bin"
pose_encoder_path = "./models/stablemakeup/pytorch_model_2.bin"

# Load base UNet
Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")

# Convert UNet to controlnets
id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)

# Load makeup encoder
makeup_encoder = detail_encoder(Unet, "./models/image_encoder_l", "cuda", dtype=torch.float32)
#makeup_encoder = detail_encoder(Unet, "openai/clip-vit-base-patch32", "cuda", dtype=torch.float32)

# Load custom weights
id_encoder.load_state_dict(torch.load(id_encoder_path), strict=False)
pose_encoder.load_state_dict(torch.load(pose_encoder_path), strict=False)
makeup_encoder.load_state_dict(torch.load(makeup_encoder_path), strict=False)

# Move models to CUDA
id_encoder.to("cuda")
pose_encoder.to("cuda")
makeup_encoder.to("cuda")

# Setup pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Inference function
def infer():
    id_folder = "./test_imgs/id"
    makeup_folder = "./test_imgs/makeup"
    out_folder = "./output"
    os.makedirs(out_folder, exist_ok=True)

    for name in os.listdir(id_folder):
        if not is_image_file(name):
            continue
        id_image = load_image(os.path.join(id_folder, name)).resize((512, 512))
        for mu in os.listdir(makeup_folder):
            if not is_image_file(mu):
                continue
            makeup_image = load_image(os.path.join(makeup_folder, mu)).resize((512, 512))
            pose_image = get_draw(id_image, size=512)

            for k in range(3):
                result_img = makeup_encoder.generate(
                    id_image=[id_image, pose_image],
                    makeup_image=makeup_image,
                    pipe=pipe,
                    guidance_scale=1.6
                )
                result_img.save(os.path.join(out_folder, f"{name.split('.')[0]}_{mu.split('.')[0]}_{k}.jpg"))

# Run
if __name__ == '__main__':
    infer()
