# =============================================================================
# FRUIT CLASSIFICATION PREDICTION SCRIPT
# =============================================================================
# This script loads your trained ResNet18 model and makes predictions on new images
# It demonstrates how to use your trained model for real-world inference

import torch
# PyTorch core library - provides tensors, neural network operations, and GPU support

from torchvision import transforms, models
# transforms: Image preprocessing operations (resize, normalize, etc.)
# models: Pre-trained neural network architectures (ResNet, etc.)

from PIL import Image
# PIL (Python Imaging Library): For opening and manipulating image files

import os
# Operating system interface - for file path operations

import numpy as np
# Numerical computing library - for array operations and calculations

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Device configuration - use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model checkpoint path - load your best fine-tuned model
MODEL_PATH = "model/best_resnet18_finetune.pt"

# Image preprocessing settings (must match what was used during training)
IMAGE_SIZE = 224  # ResNet18 expects 224x224 pixel images
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization mean values
IMAGENET_STD = [0.229, 0.224, 0.225]   # ImageNet normalization standard deviation values

# =============================================================================
# IMAGE PREPROCESSING PIPELINE
# =============================================================================
# This transform converts any input image to the format expected by the model

transform = transforms.Compose([
    # Step 1: Resize the image to 224x224 pixels (required by ResNet18)
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    # Step 2: Convert PIL image to PyTorch tensor
    # This changes format from (Height, Width, Channels) to (Channels, Height, Width)
    # and converts pixel values from [0, 255] to [0, 1]
    transforms.ToTensor(),
    
    # Step 3: Normalize the image using ImageNet statistics
    # This ensures the input distribution matches what the pre-trained model expects
    # Formula: (pixel_value - mean) / std for each color channel
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# =============================================================================
# MODEL LOADING FUNCTION
# =============================================================================

def load_model(model_path):
    """
    Load the trained model from checkpoint file
    
    Args:
        model_path (str): Path to the saved model checkpoint
        
    Returns:
        model: Loaded PyTorch model
        classes (list): List of class names in the order they were trained
    """
    print(f"Loading model from: {model_path}")
    
    # Load the checkpoint file (contains model weights and class names)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Extract class names from the checkpoint
    classes = checkpoint["classes"]
    print(f"Model was trained on classes: {classes}")
    
    # Create a new ResNet18 model with the same architecture
    model = models.resnet18(weights=None)  # Don't load ImageNet weights
    
    # Modify the final layer to match our number of classes
    num_classes = len(classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load the trained weights into the model
    model.load_state_dict(checkpoint["model_state"])
    
    # Move model to the appropriate device (GPU/CPU)
    model = model.to(DEVICE)
    
    # Set model to evaluation mode (disables dropout, batch normalization uses running stats)
    model.eval()
    
    print("Model loaded successfully!")
    return model, classes

# =============================================================================
# IMAGE PREDICTION FUNCTION
# =============================================================================

def predict_image(model, classes, image_path, transform):
    """
    Make a prediction on a single image
    
    Args:
        model: Loaded PyTorch model
        classes (list): List of class names
        image_path (str): Path to the image file
        transform: Image preprocessing pipeline
        
    Returns:
        predicted_class (str): Predicted class name
        confidence (float): Confidence score (0-1)
        all_probabilities (dict): Probability for each class
    """
    try:
        # Step 1: Open the image file using PIL
        image = Image.open(image_path)
        print(f"Loaded image: {image_path}")
        print(f"Image size: {image.size}")
        
        # Step 2: Apply preprocessing transforms
        # This converts the image to the format expected by the model
        input_tensor = transform(image)
        
        # Step 3: Add batch dimension (models expect batches, even for single images)
        # Shape changes from (3, 224, 224) to (1, 3, 224, 224)
        input_tensor = input_tensor.unsqueeze(0)
        
        # Step 4: Move tensor to the same device as the model
        input_tensor = input_tensor.to(DEVICE)
        
        # Step 5: Disable gradient computation for inference (saves memory and speeds up)
        with torch.no_grad():
            # Step 6: Forward pass through the model
            # This produces raw logits (unnormalized scores for each class)
            outputs = model(input_tensor)
            
            # Step 7: Apply softmax to convert logits to probabilities
            # Softmax ensures all probabilities sum to 1
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Step 8: Get the predicted class (index of highest probability)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            
            # Step 9: Get the confidence score (probability of predicted class)
            confidence = probabilities[0][predicted_idx].item()
            
            # Step 10: Get the predicted class name
            predicted_class = classes[predicted_idx]
            
            # Step 11: Create dictionary of all class probabilities
            all_probabilities = {}
            for i, class_name in enumerate(classes):
                all_probabilities[class_name] = probabilities[0][i].item()
        
        return predicted_class, confidence, all_probabilities
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_multiple_images(model, classes, image_folder, transform):
    """
    Make predictions on all images in a folder
    
    Args:
        model: Loaded PyTorch model
        classes (list): List of class names
        image_folder (str): Path to folder containing images
        transform: Image preprocessing pipeline
    """
    print(f"\n{'='*60}")
    print(f"PREDICTING IMAGES FROM: {image_folder}")
    print(f"{'='*60}")
    
    # Get list of image files in the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print("No image files found in the specified folder!")
        return
    
    print(f"Found {len(image_files)} images to predict")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n--- Image {i}/{len(image_files)} ---")
        
        # Make prediction
        predicted_class, confidence, all_probabilities = predict_image(
            model, classes, image_path, transform
        )
        
        if predicted_class is not None:
            # Display results
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Show all probabilities
            print("All class probabilities:")
            for class_name, prob in sorted(all_probabilities.items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        else:
            print("Failed to process this image")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Step 1: Load the trained model
    model, classes = load_model(MODEL_PATH)
    
    # Step 2: Define folder containing test images
    # You can change this path to test different images
    test_image_folder = "test_images"
    
    # Step 3: Create test folder if it doesn't exist
    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)
        print(f"\nCreated folder: {test_image_folder}")
        print("Please add some fruit images (apple, banana, mango) to this folder")
        print("Then run this script again to test predictions!")
    else:
        # Step 4: Make predictions on all images in the folder
        predict_multiple_images(model, classes, test_image_folder, transform)
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE!")
    print(f"{'='*60}")
