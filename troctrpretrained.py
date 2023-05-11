from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

from fastapi import FastAPI, File, UploadFile

# load image from the IAM database
# url = "Images/mhamed.png"
# image = Image.open(url).convert("RGB")
app = FastAPI()

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@app.post("/process_image")
async def process_image(image: UploadFile):
    image_io = BytesIO(await image.read())
    
    # Open the image using Pillow
    pillow_image = Image.open(image_io).convert("RGB")
    pixel_values = processor(images=pillow_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

    # return the OCR result
    return {"ocr_result": generated_text}
