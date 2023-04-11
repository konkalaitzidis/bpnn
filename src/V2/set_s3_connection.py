import re
import boto3



#Set up S3 connection (using default access file ~/.aws/credentials)
s3_client = boto3.client("s3", endpoint_url="https://s3.ki.se")

def generate_s3_url(path, expires_in=15*60):
    split = re.search("https://s3.ki.se/([^/]+)/(.+)", path)
    if split is None:
        raise ValueError(f"Path {path} not a valid object under https://s3.ki.se/")
    params = dict(Bucket=split[1], Key=split[2])
    return s3_client.generate_presigned_url('get_object',
                                            Params=params, 
                                            ExpiresIn=expires_in)