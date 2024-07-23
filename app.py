import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, FocalNetForImageClassification
import gradio as gr

# Path to the model
MODEL_PATH = "MichalMlodawski/nsfw-image-detection-large"

# Load the model and feature extractor
feature_extractor = AutoProcessor.from_pretrained(MODEL_PATH)
model = FocalNetForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mapping from model labels to NSFW categories
LABEL_TO_CATEGORY = {
    "LABEL_0": "Safe",
    "LABEL_1": "Questionable",
    "LABEL_2": "Unsafe"
}

def classify_image(image):
    if image is None:
        return "No image uploaded"

    # Convert to RGB (in case of PNG with alpha channel)
    image = Image.fromarray(image).convert("RGB")
    
    # Process image using feature_extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get the label from the model's configuration
    label = model.config.id2label[predicted.item()]
    category = LABEL_TO_CATEGORY.get(label, "Unknown")
    confidence_value = confidence.item() * 100

    # Prepare the result string
    emoji = {"Safe": "✅", "Questionable": "⚠️", "Unsafe": "🔞"}.get(category, "❓")
    confidence_bar = "🟩" * int(confidence_value // 10) + "⬜" * (10 - int(confidence_value // 10))
    
    result = f"{emoji} NSFW Category: {category}\n"
    result += f"🏷️ Model Label: {label}\n"
    result += f"🎯 Confidence: {confidence_value:.2f}% {confidence_bar}"

    return result

# Define Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Textbox(label="Classification Result"),
    title="🖼️ NSFW Image Classification 🔍",
    description="Upload an image to classify its safety level!",
    theme=gr.themes.Soft(primary_hue="purple"),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()