from ultralytics import YOLOv10


image = [
    rf"/mnt/d/dust/2024-06-07/44030001631310008001/44030001631310008001_2024-06-07_03-08-00_57.jpg",
    rf"/mnt/d/dust/2024-06-07/44030001631310008001/44030001631310008001_2024-06-07_09-26-11_147.jpg",
    rf"/mnt/d/dust/2024-06-07/44030001631310001031/44030001631310001031_2024-06-07_09-22-18_296.jpg",
    rf"/mnt/d/dust/2024-06-07/44030001631310001021/44030001631310001021_2024-06-07_09-19-33_295.jpg",
]

image = rf'/mnt/d/dust/2024-06-07/33052300631310016002/33052300631310016002_2024-06-07_12-09-39_21.jpg'

model = YOLOv10('yolov10x.pt')
# or
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')

option = {
    'source': image,
    'imgsz': 640,
    'device': 0,
    'save': True,
}


model.predict(**option)
