from ultralytics import YOLO

model = YOLO('runs/train/VisDrone/YOLO11n-GPSA-Neck_AvgPool-OurBlock/weights/best.pt')
 
metrics = model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", imgsz=640, batch=16, iou=0.7, device="0, 1")
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category