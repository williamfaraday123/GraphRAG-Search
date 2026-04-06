import boto3
from langchain.vectorstores import FAISS # or Weaviate, Milvus, etc.
from langchain.embeddings import OpenAIEmbeddings
from typing import Dict, Any

class ObjectStorageRetriever:
    def __init__(self, vector_store, bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, endpoint_url: Optional[str] = None):
        """
        Initialize Retriever with Vector Store and Object Storage for source fetching.
        """
        self.vector_store = vector_store
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,             # e.g., 'http://minio:9000'
            aws_access_key_id=aws_access_key_id,   # e.g., 'minioadmin'
            aws_secret_access_key=aws_secret_access_key, # e.g., 'minioadmin123'
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # Required by boto3, but ignored by MinIO
        )


    def fetch_file_content(self, s3_uri: str) -> str:
        """
        Fetches the full text content of a file from MinIO based on its URI.
        
        Args:
            s3_uri (str): The URI of the file, e.g., "s3://rag-datasets/filename.pdf"
            
        Returns:
            str: The content of the file.
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        # 1. Parse the URI to get Bucket and Key
        # Remove 's3://' and split the first slash
        path_without_protocol = s3_uri.replace("s3://", "")
        try:
            bucket, key = path_without_protocol.split("/", 1)
        except ValueError:
            raise ValueError(f"Invalid S3 URI structure (missing key): {s3_uri}")

        # 2. Download the file from MinIO
        try:
            # We use 'get_object' to stream the file content
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read the body and decode to string (assuming UTF-8 text)
            # Note: If you store PDFs/Images, you would return bytes instead.
            content = response['Body'].read().decode('utf-8')
            
            return content
            
        except Exception as e:
            # Re-raise with context so your HybridRetriever logger catches it
            raise Exception(f"Failed to fetch {s3_uri} from MinIO: {str(e)}")
            
    def get_relevant_documents(self, query: str, k: int = 3):
        """
        1. Search Vector DB for relevant snippets.
        2. (Optional) Fetch full original document from Object Storage if needed.
        """
        # Step 1: Semantic Search
        docs = self.vector_store.similarity_search(query, k=k)
        
        enriched_results = []
        
        for doc in docs:
            source_path = doc.metadata.get("source") # e.g., s3://bucket/file.txt
            
            # Step 2: Fetch full context from Object Storage (if source is S3)
            full_content = None
            if source_path and source_path.startswith("s3://"):
                s3_key = source_path.replace("s3://", "")
                try:
                    obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                    full_content = obj['Body'].read().decode('utf-8')
                except Exception as e:
                    print(f"Could not fetch source {source_path}: {e}")
            
            enriched_results.append({
                "snippet": doc.page_content,
                "full_document": full_content,
                "metadata": doc.metadata
            })

        return enriched_results

# Usage Concept
# retriever = ObjectStorageRetriever(vector_store=vs, bucket_name="...")
# results = retriever.get_relevant_documents("What is the policy on refunds?")