from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO
import time
import os


from fastapi import FastAPI, File, UploadFile

# load image from the IAM database
# url = "Images/mhamed.png"
# image = Image.open(url).convert("RGB")
app = FastAPI()


#Generating time function ms
def current_milli_time():
    return str(round(time.time() * 1000))


# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

@app.post("/process_image")
async def process_image(image: UploadFile):
    image_io = BytesIO(await image.read())
    
    # Open the image using Pillow
    pillow_image = Image.open(image_io).convert("RGB")
    pixel_values = processor(images=pillow_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

    #generating folder
    path = "data/" + generated_text + "/"
    filename = current_milli_time() + ".jpg"
    full_path = os.path.join(path, filename)

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    with open("data/"+generated_text+"/"+current_milli_time()+".jpg", "wb") as f:
        f.write(image_io.getbuffer())

    # return the OCR result
    return {"ocr_result": generated_text}
