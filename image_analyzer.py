from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np

# Load pre-trained ResNet model with weights
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels
with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

def analyze_image(image_path):
    image = Image.open(image_path)
    
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check if the image is too simple (e.g., just random shapes)
    if is_simple_image(image):
        return "This image appears to contain simple shapes or patterns and may not have significant content to analyze."

    # Transform the image
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    # Get model prediction
    with torch.no_grad():
        output = model(batch_t)

    # Get top 3 predictions
    _, indices = torch.sort(output, descending=True)
    percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100
    predictions = [(categories[idx], percentages[idx].item()) for idx in indices[0][:3]]

    # Generate description
    description = "This image likely contains: "
    for pred, score in predictions:
        description += f"{pred} ({score:.2f}% confidence), "
    description = description.rstrip(', ') + "."

    return description

def is_simple_image(image):
    # Convert image to grayscale and get unique pixel values
    gray_image = image.convert('L')
    unique_pixels = len(np.unique(np.array(gray_image)))
    
    # If there are very few unique pixel values, it's likely a simple image
    return unique_pixels < 10  # You may need to adjust this threshold
