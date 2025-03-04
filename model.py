import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import cv2  # Add this import for predict_single_image function
from preprocessing_classification2 import prepare_dataset
from sklearn.utils.class_weight import compute_class_weight

def build_model(input_shape=(64, 128, 3), num_classes=2):
    """
    Build a CNN model for text language classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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
    
    Args:
        labels: List or array of labels
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

def train_model(model, train_generator, val_generator, class_weights=None, epochs=50, batch_size=32):
    """
    Train the model with early stopping and learning rate reduction
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        class_weights: Optional dictionary of class weights to handle imbalance
        epochs: Maximum number of epochs to train
        batch_size: Batch size for training
    
    Returns:
        history: Training history
        model: Trained model
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
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights if provided
        verbose=1
    )
    
    # Save the final model
    model.save('language_classifier_model.h5')
    print("Model saved to 'language_classifier_model.h5'")
    
    return history, model

def evaluate_model(model, X_test, y_test, label_mapping):
    """
    Evaluate the model on test data and generate classification report
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (numeric)
        label_mapping: Dictionary mapping numeric labels to text labels
    """
    # Make predictions on test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Create mapping for class labels
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(label_mapping))]
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot accuracy and loss history if available
    if hasattr(model, 'history') and model.history is not None:
        plot_training_history(model.history)

def plot_training_history(history):
    """
    Plot training & validation accuracy and loss
    
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
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

def predict_single_image(model, image_path, label_mapping, target_size=(64, 128)):
    """
    Make a prediction on a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        label_mapping: Dictionary mapping numeric labels to text labels
        target_size: Size to resize image to (height, width)
    
    Returns:
        predicted_class: Predicted class label (text)
        confidence: Confidence score for the prediction
    """
    # Create inverse label mapping
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate aspect ratio
    h, w = img_rgb.shape[:2]
    aspect = w / h
    
    # Calculate new dimensions while maintaining aspect ratio
    if aspect > target_size[1] / target_size[0]:  # Wider than target
        new_width = target_size[1]
        new_height = int(new_width / aspect)
    else:  # Taller than target
        new_height = target_size[0]
        new_width = int(new_height * aspect)
        
    # Resize the image
    resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a blank target-sized image (black padding)
    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate padding to center the image
    x_offset = (target_size[1] - new_width) // 2
    y_offset = (target_size[0] - new_height) // 2
    
    # Copy the resized image onto the padded image
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Normalize pixel values to [0, 1]
    normalized = padded.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_img = np.expand_dims(normalized, axis=0)
    
    # Make prediction
    prediction = model.predict(input_img)[0]
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    predicted_class = inv_label_mapping[predicted_idx]
    
    return predicted_class, confidence

def main():
    """Main function to train and evaluate the model"""
    print("Starting language classification model training...")
    
    try:
        # Either load saved data or prepare it from scratch
        if os.path.exists("test_images.npy") and os.path.exists("test_labels.npy") and os.path.exists("label_mapping.npy"):
            print("Loading preprocessed data from disk...")
            X_test = np.load("test_images.npy")
            y_test = np.load("test_labels.npy")
            label_mapping = np.load("label_mapping.npy", allow_pickle=True).item()
            
            # Alternative approach - remove the parameter
            print("Preparing dataset generators...")
            train_generator, val_generator, (_, _), _ = prepare_dataset()  # Remove the allowed_languages parameter
        else:
            print("Preparing dataset from scratch...")
            train_generator, val_generator, (X_test, y_test), label_mapping = prepare_dataset(allowed_languages=["Arabic", "Latin"])
            train_labels = y_test  # Use test labels as proxy if we don't have actual train labels
        
        # Check class distribution in test set
        print("\nChecking test data class distribution:")
        unique_classes, class_counts = check_class_distribution(y_test)
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_test
        )
        class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}
        print(f"Computed class weights: {class_weight_dict}")
        
        # Build the model
        print("\nBuilding model...")
        model = build_model(input_shape=(64, 128, 3), num_classes=len(label_mapping))
        model.summary()
        
        # Train the model with class weights
        print("\nTraining model with class weights...")
        history, model = train_model(
            model, 
            train_generator, 
            val_generator, 
            class_weights=class_weight_dict,
            epochs=50
        )
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        evaluate_model(model, X_test, y_test, label_mapping)
        
        print("\nModel training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    main()