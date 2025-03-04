import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from pathlib import Path

# ----------------- PREPROCESSING FUNCTIONS -----------------

def load_dataset(cropped_images_dir="cropped_images", labels_dir="labels", allowed_languages=["Arabic", "Latin"]):
    """
    Load the cropped images and their corresponding language labels.
    Only includes Arabic and Latin languages.
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
    """
    unique_labels = sorted(list(set(labels)))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    
    encoded_labels = np.array([label_mapping[label] for label in labels])
    
    print(f"Label mapping: {label_mapping}")
    print(f"Label distribution: {dict(zip(*np.unique(encoded_labels, return_counts=True)))}")
    
    return encoded_labels, label_mapping

def prepare_dataset(test_size=0.2, validation_size=0.1, target_size=(128, 64), batch_size=32, allowed_languages=["Arabic", "Latin"]):
    """
    Prepare the dataset for training a classification model.
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

def visualize_samples(images, labels, label_mapping, num_samples=10):
    """
    Visualize random samples from the dataset.
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

# ----------------- MODEL FUNCTIONS -----------------

def build_model(input_shape=(64, 128, 3), num_classes=2):
    """
    Build a CNN model for text language classification
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def check_class_distribution(labels):
    """
    Check the distribution of classes in the dataset
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("\nClass Distribution:")
    print("------------------")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} samples ({count/total*100:.2f}%)")
    print("------------------\n")
    
    # Alert if significant imbalance (defined as one class having >2x samples than another)
    if max(counts) > 2 * min(counts):
        print("WARNING: Significant class imbalance detected!")
        print("Consider using class weights, data augmentation, or resampling techniques.\n")
    
    return unique_labels, counts

def train_model(model, train_generator, val_generator, class_weights=None, epochs=50):
    """
    Train the model with early stopping and learning rate reduction
    """
    # Create directories for model checkpoints
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Callbacks for training
    callbacks = [
        # Save best model
        ModelCheckpoint(
            'model_checkpoints/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping if model stops improving
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience for better convergence
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the final model
    model.save('language_classifier_model.h5')
    print("Model saved to 'language_classifier_model.h5'")
    
    return history, model

def evaluate_model(model, X_test, y_test, label_mapping):
    """
    Evaluate the model on test data and generate classification report
    """
    # Make predictions on test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Create mapping for class labels
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(label_mapping))]
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Try to plot training history if available
    try:
        if hasattr(model, 'history') and model.history is not None:
            plot_training_history(model.history)
    except Exception as e:
        print(f"Could not plot training history: {e}")

def plot_training_history(history):
    """
    Plot training & validation accuracy and loss
    """
    # Check if history object has any data
    if not hasattr(history, 'history') or not history.history:
        print("No training history to plot")
        return
        
    plt.figure(figsize=(12, 5))
    
    # Check available keys in history
    print(f"Available metrics in history: {list(history.history.keys())}")
    
    # Plot accuracy if available
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss if available
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")

# ----------------- MAIN PIPELINE -----------------

def main():
    """Main function to run the complete classification pipeline"""
    print("Starting language classification pipeline...")
    
    # Clean up any existing preprocessed data files
    for file in ["test_images.npy", "test_labels.npy", "label_mapping.npy"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed existing file: {file}")
    
    try:
        # Step 1: Prepare the dataset with only Arabic and Latin classes
        print("\n--- PREPROCESSING START ---")
        train_generator, val_generator, (X_test, y_test), label_mapping = prepare_dataset(
            allowed_languages=["Arabic", "Latin"]
        )
        print("--- PREPROCESSING COMPLETE ---\n")
        
        # Step 2: Check class distribution
        print("Checking class distribution in test set:")
        unique_classes, class_counts = check_class_distribution(y_test)
        
        # Step 3: Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_test
        )
        class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}
        print(f"Computed class weights: {class_weight_dict}")
        
        # Step 4: Build the model
        print("\n--- MODEL BUILDING START ---")
        model = build_model(input_shape=(64, 128, 3), num_classes=len(label_mapping))
        model.summary()
        print("--- MODEL BUILDING COMPLETE ---\n")
        
        # Step 5: Train the model
        print("\n--- MODEL TRAINING START ---")
        history, model = train_model(
            model, 
            train_generator, 
            val_generator, 
            class_weights=class_weight_dict,
            epochs=75  # More epochs for better convergence
        )
        print("--- MODEL TRAINING COMPLETE ---\n")
        
        # Step 6: Evaluate the model
        print("\n--- MODEL EVALUATION START ---")
        evaluate_model(model, X_test, y_test, label_mapping)
        print("--- MODEL EVALUATION COMPLETE ---\n")
        
        print("Language classification pipeline completed successfully!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full stack trace
        print(f"Error during pipeline execution: {e}")

if __name__ == "__main__":
    main()