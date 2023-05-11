import imp
import kraken.templates as tmp

# Load the trained model
model_path = 'model.mlmodel'
model = tmp.load_any(model_path)

# Read the image file
image_path = 'Images/h.jpg'
with open(image_path, 'rb') as f:
    image = f.read()

# Recognize the text in the image
text = kraken.recognize(image, model=model)

# Print the OCR output
print(text)