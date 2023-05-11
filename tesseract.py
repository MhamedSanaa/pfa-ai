import cv2
import io
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()
# Load the image using OpenCV
img = cv2.imread("Images/arab_eng.jpg")

#######################image basic preprocessing options

# Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to remove noise and make the text clearer
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

########################image advanced preprocessing options

"""gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove noise and smooth the image with a Gaussian filter
gray = cv2.GaussianBlur(gray, (3,3), 0)

# Apply adaptive thresholding to get a binary image
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)

# Apply morphological transformations to remove noise and improve character connectivity
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)"""


############################ more advanced preprocessing
# Convert the image to grayscale
def preprocess_image(image):
    # your image preprocessing code here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise and smooth the image with a median filter
    gray = cv2.medianBlur(gray, 3)

    # Threshold the image to create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove small noise and fill gaps between the elements
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Invert the image to prepare it for OCR
    binary = cv2.bitwise_not(closing)
    return binary


# Apply OCR using Tesseract
# text = pytesseract.image_to_string(binary, lang="eng")


def perform_ocr(image):
    # perform OCR using Tesseract
    ocr_result = pytesseract.image_to_string(image, lang="eng")
    return ocr_result


# Display the original and preprocessed images, and the recognized text
"""fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[1].imshow(binary, cmap="gray")
axs[1].set_title("Preprocessed Image")
axs[2].text(0.1, 0.5, text, fontsize=12)
axs[2].axis("off")
axs[2].set_title("Recognized Text")
plt.show()"""


@app.post("/process_image")
async def process_image(image: UploadFile):
    # read the uploaded image file
    image_content = await image.read()
    # img = Image.open(io.BytesIO(image_content))

    np_image = np.fromstring(image_content, np.uint8)
    cv_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # preprocess the image
    processed_image = preprocess_image(cv_image)

    # perform OCR on the processed image
    ocr_result = perform_ocr(processed_image)

    # return the OCR result
    return {"ocr_result": ocr_result}
