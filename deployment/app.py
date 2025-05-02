from flask import Flask, render_template, request, jsonify
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

# Import your model definition
from model import CNNModel

app = Flask(__name__)

# Model settings
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "logs-augmented",
    "cnn_lr0.0001_dr0.8_bs64_mp2_model.pth",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Pork", "Mix", "Beef"]
IMAGE_SIZE = 112  # Your CNN uses 112x112 images


# Load model
def load_model():
    model = CNNModel(dropout_rate=0.8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    return model


# Initialize model
model = load_model()


# Image preprocessing - using the same transforms as in training
def preprocess_image(image_bytes):
    # Define the same transforms you used during training
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5484, 0.3619, 0.3821],  # Same as in train.py for CNN
                std=[0.1129, 0.1049, 0.1092],  # Same as in train.py for CNN
            ),
        ]
    )

    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert grayscale to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/api/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    img_bytes = file.read()

    try:
        # Preprocess the image
        image_tensor = preprocess_image(img_bytes)

        # Move tensor to device
        image_tensor = image_tensor.to(DEVICE)

        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"outputs: {outputs}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            print(f"probabilities: {probabilities}")

        # Convert to Python list
        probs = probabilities.cpu().numpy().tolist()
        print(f"probs: {probs}")

        # Get class probabilities
        pork_prob = float(probs[0])  # Convert numpy float32 to Python float
        mix_prob = float(probs[1])
        beef_prob = float(probs[2])

        # Get predicted class
        predicted_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_idx]

        return jsonify(
            {
                "success": True,
                "probabilities": {
                    "beef": beef_prob,
                    "pork": pork_prob,
                    "mix": mix_prob,
                },
                "predicted_class": predicted_class,
            }
        )

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
