from ultralytics import YOLOv10

model = YOLOv10('yolov10l.pt')
# or
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')

option = {
    'format': 'onnx',
    'imgsz': 640,
    'batch': 1,
    'simplify': True,
    'nms': True,
}

model.export(**option)