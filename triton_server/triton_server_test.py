import os
from pathlib import Path
import shutil
from ultralytics import YOLOv10


class TritonServer:
    
    def __init__(self, model_size: str, triton_model_repo: str):
        self.model_size = model_size
        self.triton_model_repo = Path(triton_model_repo)
        if not self.triton_model_repo.exists():
            self.triton_model_repo.mkdir(exist_ok=True, parents=True)
        self.config_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.yolo_path = self.config_path.parent
        self.config_file_name = 'config.pbtxt'
        self.onnx_model_file_name = 'model.onnx'
        
    def export(self):
        model = YOLOv10(self.yolo_path.joinpath(f'{self.model_size}.pt'))
        option = {
            'format': 'onnx',
            'imgsz': 640,
            'batch': 1,
            'simplify': True,
            # 'nms': True,
        }
        model.export(**option)
    
    def mkdir(self, version='1'):
        self.yolo_triton_path = self.triton_model_repo.joinpath(self.model_size)
        if self.yolo_triton_path.exists():
            shutil.rmtree(self.yolo_triton_path.absolute().as_posix())
        self.yolo_triton_path_version = self.triton_model_repo.joinpath(self.model_size).joinpath(version)
        self.yolo_triton_path_version.mkdir(parents=True)
    
    def cp_file(self):
        onnx_file = self.yolo_path.joinpath(f'{self.model_size}.engine')
        shutil.copyfile(onnx_file, self.yolo_triton_path_version.joinpath(self.onnx_model_file_name))
        
        from string import Template
        template_file = self.config_path.joinpath('config-template.pbtxt')

        with open(template_file, mode='r') as f:
            template = Template(f.read())
        config_content = template.substitute(model_size=self.model_size)
        
        config_file = self.yolo_triton_path.joinpath(self.config_file_name)
        with open(config_file, mode='w') as f:
            f.write(config_content)
        
    def start_docker(self):
        import contextlib
        import subprocess
        import time
        from tritonclient.http import InferenceServerClient

        container_id = (
            subprocess.check_output(
                        f"docker run -d --gpus=1 --rm --net=host -v {self.triton_model_repo}:/models nvcr.io/nvidia/tritonserver:23.10-py3 --model-repository=/models",
                shell=True,
            ).decode("utf-8")
             .strip()
        )
        # Wait for the Triton server to start
        triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
        # Wait until model is ready
        for _ in range(10):
            with contextlib.suppress(Exception):
                assert triton_client.is_model_ready(self.model_size)
                break
            time.sleep(1)
        return container_id
        
    def call_http_interface(self, server=f'http://localhost:8000', task='detect', test_img: str=None):
        model = YOLOv10(f'{server}/{self.model_size}', task=task)
        assert test_img is not None
        option = {
            'source': test_img,
            # 'imgsz': 640,
            # 'device': 0,
            'save': True,
        }
        results = model(**option)
        print(results)

    def clean_docker(self, container_id: str):
        import subprocess
        subprocess.call(f"docker kill {container_id}", shell=True)


if __name__ == '__main__':
    
    server = TritonServer(
        model_size='yolov10l',
        triton_model_repo=rf'/home/zhangningboo/workspace/yolov10/triton_server/model_repository'
    )
    server.export()
    server.mkdir()
    server.cp_file()
    try:
        container_id = server.start_docker()
        server.call_http_interface(
            server=f'http://localhost:8000',
            test_img=rf"/home/zhangningboo/Pictures/33108200631320002004/33108200631320002004_2024-05-24_2.jpg"
        )
    except Exception as e:
        print(e)
    finally:
        if container_id:
            server.clean_docker(container_id)
