"""Upload all .txt files from raw_data folder to MinIO's rag-datasets bucket."""
import os
import glob
from data_loader import MinIODataLoader

def upload_raw_data():
    loader = MinIODataLoader()
    
    # Find all .txt files in the raw_data folder (and subfolders)
    raw_dir = os.getenv("DATA_SOURCE_PATH", "./raw_data")
    files = glob.glob(f"{raw_dir}/**/*.txt", recursive=True)
    
    if not files:
        print(f"No .txt files found in '{raw_dir}'.")
        print(f"Place your .txt files in the '{raw_dir}' folder and run again.")
        return
    
    for filepath in files:
        loader.upload_file(filepath)
    
    print(f"\nUploaded {len(files)} file(s). You can now run: python main.py")

if __name__ == "__main__":
    upload_raw_data()
