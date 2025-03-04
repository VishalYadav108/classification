import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(cropped_images_dir="cropped_images", labels_dir="labels", allowed_languages=["Arabic", "Latin"]):
    """
    Load the cropped images and their corresponding language labels.
    
    Args:
        cropped_images_dir: Directory containing cropped images
        labels_dir: Directory containing language label text files
        allowed_languages: List of language labels to include (filter out others)
    
    Returns:
        images: List of loaded images
        labels: List of corresponding labels
        filenames: List of image filenames (without extension)
    """
    images = []
    labels = []
    filenames = []
    
    cropped_dir = Path(cropped_images_dir)
    labels_dir = Path(labels_dir)
    
    if not cropped_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Either {cropped_dir} or {labels_dir} does not exist")
    
    # Get all image files
    image_files = list(cropped_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} total images")
    skipped_count = 0
    
    for img_path in image_files:
        filename = img_path.stem
        label_path = labels_dir / f"{filename}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file found for {filename}")
            skipped_count += 1
            continue
        
        # Read the label first to check if it should be included
        with open(label_path, 'r') as f:
            label = f.read().strip()
        
        # Skip if not in allowed languages
        if allowed_languages and label not in allowed_languages:
            skipped_count += 1
            continue
            
        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            skipped_count += 1
            continue
            
        # Store the data
        images.append(img)
        labels.append(label)
        filenames.append(filename)
    
    print(f"Kept {len(images)} images ({skipped_count} skipped due to invalid labels or missing files)")
    
    return images, labels, filenames

def preprocess_images(images, target_size=(128, 64)):
    """
    Preprocess images by resizing them to a standard size while maintaining aspect ratio.
    
    Args:
        images: List of images to preprocess
        target_size: Target image size (width, height)
        
    Returns:
        preprocessed_images: Numpy array of preprocessed images
    """
    preprocessed_images = []
    
    for img in images:
        # Convert to RGB if needed (OpenCV loads as BGR)
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
        
        preprocessed_images.append(normalized)
    
    return np.array(preprocessed_images)

def encode_labels(labels):
    """
    Convert text labels to numerical values.
    
    Args:
        labels: List of textual labels
        
    Returns:
        encoded_labels: Numpy array of encoded labels (0 for Latin, 1 for Arabic)
        label_mapping: Dictionary mapping numerical values back to text labels
    """
    unique_labels = sorted(list(set(labels)))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    
    encoded_labels = np.array([label_mapping[label] for label in labels])
    
    print(f"Label mapping: {label_mapping}")
    print(f"Label distribution: {dict(zip(*np.unique(encoded_labels, return_counts=True)))}")
    
    return encoded_labels, label_mapping

def visualize_samples(images, labels, label_mapping, num_samples=10):
    """
    Visualize random samples from the dataset.
    
    Args:
        images: Preprocessed images
        labels: Encoded labels
        label_mapping: Dictionary mapping numerical values to text labels
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create a mapping from numerical values to text labels
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(f"Class: {inv_label_mapping[labels[idx]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("sample_preprocessed_images.png")
    plt.close()
    print("Sample visualization saved as 'sample_preprocessed_images.png'")

def prepare_dataset(test_size=0.2, validation_size=0.1, target_size=(128, 64), batch_size=32, allowed_languages=None):
    """
    Prepare the dataset for training a classification model.
    
    Args:
        test_size: Fraction of data to use for testing
        validation_size: Fraction of training data to use for validation
        target_size: Target image size (width, height)
        batch_size: Batch size for data generators
        allowed_languages: List of languages to include (filter out others)
    
    Returns:
        train_data: Training data generator
        val_data: Validation data generator
        test_data: Test data (images, labels)
        label_mapping: Dictionary mapping numerical values to text labels
    """
    # Load the dataset
    print("Loading dataset...")
    images, labels, filenames = load_dataset(allowed_languages=allowed_languages)
    
    # Preprocess the images
    print("Preprocessing images...")
    preprocessed_images = preprocess_images(images, target_size)
    
    # Encode the labels
    print("Encoding labels...")
    encoded_labels, label_mapping = encode_labels(labels)
    
    # Visualize some samples
    print("Visualizing samples...")
    visualize_samples(preprocessed_images, encoded_labels, label_mapping)
    
    # Split into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_images, encoded_labels, 
        test_size=test_size, 
        random_state=42,
        stratify=encoded_labels  # Ensure balanced classes
    )
    
    # Further split training into training and validation
    if validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=validation_size / (1 - test_size),
            random_state=42,
            stratify=y_train  # Ensure balanced classes
        )
        
        # Create data generator with augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Text shouldn't be flipped horizontally
            brightness_range=(0.8, 1.2)
        )
        
        # Create data generator without augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, 
            tf.keras.utils.to_categorical(y_train, num_classes=len(label_mapping)),
            batch_size=batch_size
        )
        
        val_generator = val_datagen.flow(
            X_val, 
            tf.keras.utils.to_categorical(y_val, num_classes=len(label_mapping)),
            batch_size=batch_size
        )
        
        print(f"Dataset split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
        
        return train_generator, val_generator, (X_test, y_test), label_mapping
    else:
        # Create data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Text shouldn't be flipped horizontally
            brightness_range=(0.8, 1.2)
        )
        
        # Create generator
        train_generator = train_datagen.flow(
            X_train, 
            tf.keras.utils.to_categorical(y_train, num_classes=len(label_mapping)),
            batch_size=batch_size
        )
        
        print(f"Dataset split: {len(X_train)} training, {len(X_test)} test samples")
        
        return train_generator, None, (X_test, y_test), label_mapping
    
def main():
    """Main function to execute the script"""
    print("Starting image preprocessing for language classification...")
    
    try:
        # Explicitly pass the allowed_languages parameter to filter out unwanted classes
        train_generator, val_generator, test_data, label_mapping = prepare_dataset(allowed_languages=["Arabic", "Latin"])
        
        print("\nPreprocessing completed successfully!")
        print(f"Number of classes: {len(label_mapping)}")
        print(f"Class mapping: {label_mapping}")
        
        # Save test data and label mapping for later use
        np.save("test_images.npy", test_data[0])
        np.save("test_labels.npy", test_data[1])
        np.save("label_mapping.npy", label_mapping)
        
        print("Test data and label mapping saved to disk")
        print("Dataset is ready for model training")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()