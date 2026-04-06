import os
import boto3
from botocore.client import Config

class MinIODataLoader:
    def __init__(self):
        self.bucket_name = "rag-datasets"
        
        # 1. Connect to MinIO (Running in Docker)
        # Note: We use 'http://minio:9000' because Docker Compose DNS resolves 'minio'
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',  # <--- CRITICAL: Points to local container
            aws_access_key_id='minioadmin',    # Your MINIO_ROOT_USER
            aws_secret_access_key='minioadmin123', # Your MINIO_ROOT_PASSWORD
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # MinIO is region-agnostic, but boto3 requires a string
        )

    def upload_file(self, file_path, object_name=None):
        """Upload a file from the host to MinIO"""
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        try:
            # Ensure bucket exists (MinIO doesn't auto-create buckets like S3 sometimes does)
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except:
            self.s3_client.create_bucket(Bucket=self.bucket_name)

        # Upload
        self.s3_client.upload_file(file_path, self.bucket_name, object_name)
        print(f"Uploaded {file_path} to MinIO bucket '{self.bucket_name}'")

    def load_documents(self):
        """List and process files from MinIO"""
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        
        if 'Contents' not in response:
            print("No files found in bucket.")
            return

        for obj in response['Contents']:
            key = obj['Key']
            print(f"Processing: {key}")
            # Add your LangChain loading logic here (downloading stream to memory)