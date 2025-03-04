import os
import cv2
import numpy as np
from pathlib import Path

def create_output_folders():
    """Create necessary folders for output data"""
    os.makedirs("cropped_images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    print("Created output folders: cropped_images and labels")

def parse_text_file(text_file_path):
    """Parse text file to extract bounding box coordinates and language labels"""
    with open(text_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    regions = []
    for line in lines:
        # Skip empty lines or comment lines
        if not line.strip() or line.strip().startswith('//'):
            continue
        
        # Split by comma
        parts = line.strip().split(',')
        
        # Need at least 10 parts (8 coordinates + language + text)
        if len(parts) < 10:
            continue
            
        try:
            # Extract coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
            coords = [int(parts[i]) for i in range(8)]
            # Extract language
            language = parts[8]
            # Extract text (remaining parts joined with commas)
            text = ','.join(parts[9:])
            
            regions.append({
                'coords': coords,
                'language': language,
                'text': text
            })
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line.strip()[:50]}... Error: {e}")
            continue
            
    return regions

def crop_with_perspective_transform(image, coords):
    """Crop an image region using perspective transform"""
    # Extract corner coordinates
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    
    # Source points (quadrilateral in the original image)
    src_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    
    # Calculate width and height for the output rectangle
    width_1 = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    width_2 = np.sqrt(((x3 - x4) ** 2) + ((y3 - y4) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((x4 - x1) ** 2) + ((y4 - y1) ** 2))
    height_2 = np.sqrt(((x3 - x2) ** 2) + ((y3 - y2) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    # Ensure valid dimensions
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Invalid crop dimensions calculated")
    
    # Destination points (rectangle)
    dst_pts = np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype=np.float32)
    
    # Compute perspective transform matrix and warp the image
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return result

def process_dataset(images_folder="images", texts_folder="texts"):
    """Process all text files and their corresponding images"""
    texts_dir = Path(texts_folder)
    images_dir = Path(images_folder)
    
    # Check if folders exist
    if not texts_dir.exists():
        print(f"Error: Texts folder not found: {texts_dir}")
        return
        
    if not images_dir.exists():
        print(f"Error: Images folder not found: {images_dir}")
        return
    
    # Get all text files
    text_files = list(texts_dir.glob("*.txt"))
    
    if not text_files:
        print(f"No text files found in {texts_dir}")
        return
        
    print(f"Found {len(text_files)} text files to process")
    processed_count = 0
    
    for text_file in text_files:
        # Get base name without extension
        base_name = text_file.stem.lower()  # Convert to lowercase for case-insensitive matching
        
        # Look for corresponding image file with different extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            # Try different case variations
            for name_variant in [base_name, base_name.upper(), base_name.lower()]:
                potential_image = images_dir / f"{name_variant}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            if image_file:
                break
        
        if not image_file:
            print(f"No matching image found for {text_file}")
            continue
        
        print(f"Processing: {base_name}")
        
        # Load the image
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Failed to load image: {image_file}")
            continue
            
        # Parse the text file
        regions = parse_text_file(text_file)
        
        if not regions:
            print(f"No valid regions found in {text_file}")
            continue
            
        print(f"  Found {len(regions)} text regions to crop")
        
        # Process each region
        for i, region in enumerate(regions):
            try:
                # Crop the image region
                cropped = crop_with_perspective_transform(img, region['coords'])
                
                # Save the cropped image
                crop_filename = f"{base_name}_{i}.jpg"
                cv2.imwrite(os.path.join("cropped_images", crop_filename), cropped)
                
                # Save the language label
                with open(os.path.join("labels", f"{base_name}_{i}.txt"), 'w') as label_file:
                    label_file.write(region['language'])
                
                processed_count += 1
                print(f"    Processed region {i} - Language: {region['language']}")
                
            except Exception as e:
                print(f"    Error processing region {i} from {base_name}: {e}")
    
    print(f"Completed processing. Created {processed_count} cropped images with labels.")
                
def main():
    """Main function to execute the program"""
    print("Starting classification dataset creation...")
    create_output_folders()
    
    # Try to find the folder structure
    if os.path.exists("image50/texts"):
        print("Using folder structure: image50/")
        process_dataset(images_folder="image50/images", texts_folder="image50/texts")
    elif os.path.exists("texts") and os.path.exists("images"):
        print("Using folder structure: current directory")
        process_dataset()
    else:
        print("Could not find expected folder structure.")
        print("Please ensure you have either:")
        print("  - 'texts' and 'images' folders in the current directory, or")
        print("  - 'image50/texts' and 'image50/images' folders")
    
if __name__ == "__main__":
    main()