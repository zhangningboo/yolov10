from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
import cv2
from ultralytics import YOLOv10
import logging

logger = logging.getLogger(__name__)



class ImageReceiver(BaseHTTPRequestHandler):
    
    def __init__(self, request, client_address, server):
        self.model = YOLOv10(f'http://127.0.0.1:8000/yolov10l', task='detect')
        logger.info('模型加载完成，启动http服务...')
        
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)  # 自定义代码在上面编写，这一行是 __init__ 的最后一行
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        np_image = np.frombuffer(body, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        option = {
            'source': image,
            'conf': 0.65,
        }
        result = self.model(**option)[0]
        result.plot()
        containPeople = 0 in result.boxes.cls
        data = {
            "containPeople": containPeople
        }
        json_data = json.dumps(data).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json_data)
        

def run(server_class=HTTPServer, handler_class=ImageReceiver, port=7474):
    server = ('0.0.0.0', port)
    httpd = server_class(server, handler_class)
    httpd.serve_forever()


if __name__ == '__main__':
    run()