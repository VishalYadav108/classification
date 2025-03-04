import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse
from pathlib import Path

def preprocess_image(img_path, target_size=(128, 64)):
    """
    Preprocess a single image for prediction
    
    Args:
        img_path: Path to the image file
        target_size: Target image size (width, height)
        
    Returns:
        preprocessed_img: Normalized and resized image ready for prediction
    """
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate aspect ratio
    h, w = img_rgb.shape[:2]
    aspect = w / h
    
    # Calculate new dimensions while maintaining aspect ratio
    if aspect > target_size[0] / target_size[1]:  # Wider than target
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:  # Taller than target
        new_height = target_size[1]
        new_width = int(new_height * aspect)
        
    # Resize the image
    resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a blank target-sized image (black padding)
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate padding to center the image
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    # Copy the resized image onto the padded image
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Normalize pixel values to [0, 1]
    normalized = padded.astype(np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

def predict_image(model_path, img_path, show_visualization=True):
    """
    Make a prediction on a single image
    
    Args:
        model_path: Path to the saved model
        img_path: Path to the image file
        show_visualization: Whether to show the image with prediction
        
    Returns:
        predicted_class: Predicted class (Arabic or Latin)
        confidence: Confidence score for the prediction
    """
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_img)[0]
    
    # Get the predicted class
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    
    # Class mapping (based on your training)
    label_mapping = {0: 'Arabic', 1: 'Latin'}
    predicted_class = label_mapping[predicted_idx]
    
    # Print results
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Visualize if requested
    if show_visualization:
        # Load the original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
        plt.axis('off')
        
        # Save the visualization
        output_path = f"prediction_{os.path.basename(img_path)}"
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved as '{output_path}'")
    
    return predicted_class, confidence

def batch_predict(model_path, image_dir, output_csv=None):
    """
    Make predictions on a batch of images in a directory
    
    Args:
        model_path: Path to the saved model
        image_dir: Directory containing images to predict
        output_csv: Optional path to save results as CSV
    """
    # Load the model
    model = load_model(model_path)
    
    # Get all image files
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Class mapping
    label_mapping = {0: 'Arabic', 1: 'Latin'}
    
    # Prepare results
    results = []
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_path in image_files:
        try:
            # Preprocess the image
            preprocessed_img = preprocess_image(str(img_path))
            
            # Make prediction
            prediction = model.predict(preprocessed_img, verbose=0)[0]
            
            # Get the predicted class
            predicted_idx = np.argmax(prediction)
            confidence = prediction[predicted_idx]
            predicted_class = label_mapping[predicted_idx]
            
            # Store result
            results.append({
                'image': str(img_path),
                'prediction': predicted_class,
                'confidence': confidence
            })
            
            print(f"{img_path.name}: {predicted_class} (Confidence: {confidence:.2%})")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results to CSV if requested
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['image', 'prediction', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
            print(f"Results saved to {output_csv}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict language (Arabic or Latin) from text images')
    parser.add_argument('--model', default='language_classifier_model.h5', help='Path to the trained model')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single image prediction
    single_parser = subparsers.add_parser('single', help='Predict a single image')
    single_parser.add_argument('image', help='Path to the image file')
    single_parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict multiple images')
    batch_parser.add_argument('directory', help='Directory containing images')
    batch_parser.add_argument('--output', help='Output CSV file path')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        predict_image(args.model, args.image, not args.no_viz)
    elif args.command == 'batch':
        batch_predict(args.model, args.directory, args.output)
    else:
        # No command specified, show help
        parser.print_help()