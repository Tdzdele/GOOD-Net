from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/VisDrone/GOOD-Net-n/weights/best.pt')
    model.predict(source='datasets/VisDrone/VisDrone2019-DET-test-challenge/images',
                  imgsz=640,
                  project='runs/detect/test-challenge',
                  name='GOOD-Net-n',
                  save=True,
                  show_labels=False,
                )