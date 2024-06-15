import requests
from io import BytesIO
import cv2



http_server = rf'http://127.0.0.1:7474'

image = cv2.imread(rf'/home/aky-ubuntu/workspace/yolov10/1.jpg')
_, image_encode = cv2.imencode('.jpg', image)
image_bytes = BytesIO(image_encode)

response = requests.post(url=http_server, data=image_bytes.getvalue(), headers={'Content-Type': 'image/jpeg'})
response_data = response.json()
print(response.status_code)

