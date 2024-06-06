import os
from pathlib import Path
import shutil
from ultralytics import YOLO


class TritonServer:
    
    def __init__(self, triton_model_repo: str):
        self.triton_model_repo = Path(triton_model_repo)
        if not self.triton_model_repo.exists():
            self.triton_model_repo.mkdir(exist_ok=True, parents=True)
        self.config_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.yolo_path = self.config_path.parent
        self.config_file_name = 'config.pbtxt'
        self.onnx_model_file_name = 'model.onnx'
    
    def mkdir(self, service_name='yolov10', version='1'):
        self.yolo_triton_path = self.triton_model_repo.joinpath(service_name)
        if self.yolo_triton_path.exists():
            shutil.rmtree(self.yolo_triton_path.absolute().as_posix())
        self.yolo_triton_path_version = self.triton_model_repo.joinpath(service_name).joinpath(version)
        self.yolo_triton_path_version.mkdir(parents=True)
    
    def cp_file(self, onnx_name='yolov10l.onnx'):
        onnx_file = self.yolo_path.joinpath(onnx_name)
        shutil.copyfile(onnx_file, self.yolo_triton_path_version.joinpath(self.onnx_model_file_name))
        
        config_file = self.config_path.joinpath(self.config_file_name)
        shutil.copyfile(config_file, self.yolo_triton_path.joinpath(self.config_file_name))
        
    def call_interface(self, server=f'http://192.168.1.48:8000/yolov10', task='detect'):
        # Load the Triton Server model
        model = YOLO(f'http://192.168.1.48:8000/yolov10', task=task)
        # Run inference on the server
        results = model("/home/zhangningboo/Pictures/33108200631320002004/33108200631320002004_2024-05-24_2.jpg")
        print(results)


if __name__ == '__main__':
    
    server = TritonServer(triton_model_repo=rf'/home/zhangningboo/workspace/suzhou-master/triton-server/model_repository')
    server.mkdir()
    server.cp_file()
    server.call_interface()
