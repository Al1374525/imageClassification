from minio import Minio

#MinIO  endpoint, access key and secret key
endpoint = "localhost:9000"
access_key = "minioadmin"
secret_key = "minioadmin"

#Create MinIO client
client = Minio(endpoint, access_key, secret_key, secure=False)

#Create bucket
bucket_name = "image-classification"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"Bucket {bucket_name} created successfully.")
else:
    print(f"Bucket {bucket_name} already exists.")