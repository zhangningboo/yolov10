from ultralytics import YOLOv10


image = rf'/home/zhangningboo/Pictures/33108200631320002004/33108200631320002004_2024-05-24_2.jpg'

model = YOLOv10('yolov10l.pt')
# or
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')

option = {
    'source': image,
    'imgsz': 640,
    'device': 0,
    'save': True,
}


model.predict(**option)