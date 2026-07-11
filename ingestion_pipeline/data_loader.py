import os
import boto3
import logging
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class MinIODataLoader:
    def __init__(self):
        self.bucket_name = os.getenv("MINIO_BUCKET", "rag-datasets")
        
        # Use environment variables so it works both in Docker and locally
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_user = os.getenv("MINIO_ROOT_USER", "minioadmin")
        minio_password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_user,
            aws_secret_access_key=minio_password,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create the bucket if it doesn't already exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' already exists.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"Bucket '{self.bucket_name}' not found. Creating it...")
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                logger.warning(f"Unexpected error checking bucket: {e}")

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
        """List and download text files from MinIO, returning them as document dicts."""
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        
        docs = []
        if 'Contents' not in response:
            print("No files found in bucket.")
            return docs

        for obj in response['Contents']:
            key = obj['Key']
            print(f"Processing: {key}")
            try:
                file_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                content = file_obj['Body'].read().decode('utf-8')
                docs.append({"content": content, "source": key})
            except Exception as e:
                print(f"Failed to download {key}: {e}")
        return docs