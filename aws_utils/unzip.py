import boto3
import os
import zipfile
from tqdm import tqdm

# AWS S3 Configuration
BUCKET_NAME = "your-s3-bucket-name"
ZIP_FILE_KEY = "path/to/your/file.zip"  # Path to the ZIP file in S3
TEMP_ZIP_FILE = "/tmp/temp_file.zip"   # Temporary file storage location
UNZIPPED_DIR = "/tmp/unzipped_files"   # Temporary directory for unzipped content
DESTINATION_PREFIX = "unzipped/"       # Prefix where unzipped files will be stored in S3

# Initialize S3 Client
s3 = boto3.client('s3')

def download_and_unzip():
    # Step 1: Download ZIP File from S3 with Progress Bar
    print("Downloading the ZIP file from S3...")
    object_size = s3.head_object(Bucket=BUCKET_NAME, Key=ZIP_FILE_KEY)['ContentLength']

    with tqdm(total=object_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        s3.download_file(Bucket=BUCKET_NAME, Key=ZIP_FILE_KEY, Filename=TEMP_ZIP_FILE, Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    
    # Step 2: Unzip the File Locally with Progress Bar
    print("\nUnzipping the file...")
    os.makedirs(UNZIPPED_DIR, exist_ok=True)
    with zipfile.ZipFile(TEMP_ZIP_FILE, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Unzipping"):
            zip_ref.extract(file, UNZIPPED_DIR)
    
    # Step 3: Upload Unzipped Files Back to S3 with Progress Bar
    print("\nUploading unzipped files back to S3...")
    for root, _, files in os.walk(UNZIPPED_DIR):
        for file in tqdm(files, desc="Uploading", unit="file"):
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(DESTINATION_PREFIX, os.path.relpath(local_file_path, UNZIPPED_DIR))
            
            # Upload each file to S3
            s3.upload_file(local_file_path, BUCKET_NAME, s3_key)

    # Step 4: Clean Up Temporary Files
    print("\nCleaning up temporary files...")
    os.remove(TEMP_ZIP_FILE)
    for file in os.listdir(UNZIPPED_DIR):
        os.remove(os.path.join(UNZIPPED_DIR, file))
    print("Process completed successfully!")

if __name__ == "__main__":
    download_and_unzip()
