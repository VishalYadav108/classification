# my-python-app/README.md

# My Python App

This project is designed to separate text files and image files from a specified folder. It organizes the files into two newly created folders named "texts" and "images".

## Project Structure

```
my-python-app
├── main.py
├── README.md
└── requirements.txt
```

## Files

- **main.py**: Contains the main logic for the program. It defines a function that scans a specified folder for files, separates them into text files and image files, and saves them into newly created folders named "texts" and "images".
  
- **requirements.txt**: Lists the external libraries required for the project, such as `os` and `shutil`, which are used for file operations.

## Setup

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required libraries listed in `requirements.txt` using the following command:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open `main.py` and specify the folder path you want to scan for files.
2. Run the program:

   ```
   python main.py
   ```

3. The program will create two folders, "texts" and "images", and will move the respective files into these folders.

## Dependencies

- Python 3.x
- `os`
- `shutil`