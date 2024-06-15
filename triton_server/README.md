docker run -d --gpus=1 --rm --net=host -v /home/aky-ubuntu/workspace/yolov10/triton_server/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models

/usr/src/tensorrt/bin/trtexec --onnx=yolov10l.onnx --explicitBatch --saveEngine=yolov10l.plan