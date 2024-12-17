import boto3
import requests
import math
import os

# AWS S3 Configuration
BUCKET_NAME = "your-s3-bucket-name"
OBJECT_KEY = "large-files/file.zip"
FILE_URL = "https://example.com/largefile.zip"
MIN_PART_SIZE = 5 * 1024 * 1024  # 5 MB minimum part size

# Initialize S3 client
s3 = boto3.client('s3')

def download_and_upload():
    # Get file size from the URL headers
    response_head = requests.head(FILE_URL)
    file_size = int(response_head.headers.get('Content-Length', 0))

    if file_size == 0:
        raise ValueError("File size could not be determined. Check the URL.")

    # Calculate dynamic part size
    part_size = max(MIN_PART_SIZE, math.ceil(file_size / 10000))
    print(f"File Size: {file_size / (1024**3):.2f} GB, Part Size: {part_size / (1024**2):.2f} MB")

    # Start multipart upload
    multipart_upload = s3.create_multipart_upload(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
    upload_id = multipart_upload['UploadId']
    parts = []

    try:
        print("Starting download and multipart upload...")
        response = requests.get(FILE_URL, stream=True)
        part_number = 1

        for chunk in response.iter_content(chunk_size=part_size):
            if not chunk:
                break

            # Write chunk to a temporary file
            temp_file = f"part_{part_number}.tmp"
            with open(temp_file, 'wb') as f:
                f.write(chunk)

            # Upload part to S3
            print(f"Uploading part {part_number}...")
            with open(temp_file, 'rb') as f:
                part = s3.upload_part(
                    Bucket=BUCKET_NAME,
                    Key=OBJECT_KEY,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=f
                )
                parts.append({"PartNumber": part_number, "ETag": part['ETag']})

            # Remove the temporary file
            os.remove(temp_file)
            part_number += 1

        # Complete the multipart upload
        print("Completing multipart upload...")
        s3.complete_multipart_upload(
            Bucket=BUCKET_NAME,
            Key=OBJECT_KEY,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts}
        )
        print(f"File successfully uploaded to {BUCKET_NAME}/{OBJECT_KEY}")

    except Exception as e:
        print(f"An error occurred: {e}")
        s3.abort_multipart_upload(Bucket=BUCKET_NAME, Key=OBJECT_KEY, UploadId=upload_id)

if __name__ == "__main__":
    download_and_upload()
