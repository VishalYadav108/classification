import os
import shutil

def separate_files(folder_path):
    # Create directories for texts and images if they don't exist
    texts_folder = os.path.join(folder_path, 'texts')
    images_folder = os.path.join(folder_path, 'images')
    
    os.makedirs(texts_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # Define file extensions for text and image files
    text_extensions = ('.txt', '.md', '.csv', '.json', '.xml')
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

    # Iterate through the files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            # Separate files based on their extensions
            if filename.endswith(text_extensions):
                shutil.move(file_path, os.path.join(texts_folder, filename))
            elif filename.endswith(image_extensions):
                shutil.move(file_path, os.path.join(images_folder, filename))

if __name__ == "__main__":
    # Specify the folder to scan
    folder_to_scan = input("Enter the path of the folder to scan: ")
    separate_files(folder_to_scan)