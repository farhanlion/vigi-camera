import cloudinary.uploader
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

class Uploader():
    def __init__(self):
        self.url = 'http://localhost:8084/movementDetected'
        self.postdata = {}

        env_path = Path('.', '.env')
        load_dotenv(dotenv_path=env_path)
        cloudinary.config( 
            cloud_name = os.getenv('cloud_name'), 
            api_key =os.getenv('api_key'), 
            api_secret = os.getenv('api_secret') 
        )
    

    def upload(self,path, timestamp,movements):
        print("uploading...\n")
        result = cloudinary.uploader.upload(path, 
            resource_type = "video",
            folder = "movements",
            quality= 100
        )
        print('upload Successful\n')
        # print(result,'\n')
        self.postdata["videolink"] = result['secure_url']
        self.postdata["timestamp"] = timestamp.strftime("%m/%d/%Y, %H:%M:%S")
        self.postdata["movements"] = movements
        self.postdata["public_id"] = result['public_id']
        print('data:')
        print(self.postdata,'\n')
        print('sending to server...\n')
        self.pingWebsite()
    
    def pingWebsite(self):
        try:
            post_response = requests.post(self.url, json=self.postdata)
            post_response_json = post_response.json()
            print(post_response_json)

        except Exception as e:
            print('sending to server failed:\n')
            print(e)
        return