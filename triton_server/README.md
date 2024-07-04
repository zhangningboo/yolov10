docker run -d --gpus=1 --rm --net=host -v /home/aky-ubuntu/workspace/yolov10/triton_server/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
docker run --gpus=all --rm --net=host -v /home/aky-ubuntu/workspace/yolov10/triton_server/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models

docker run -itd --gpus=all --rm --net=host -v /home/aky-ubuntu/workspace/yolov10:/opt/yolov10 -v /home/aky-ubuntu/workspace/yolov10/triton_server/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
docker exec -it f31271117833070fb5b78bd13329afd128fa9cd8b64a32cfe43adbf19d4e82f7 bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov10l.onnx --explicitBatch --saveEngine=model.plan