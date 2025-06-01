import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('ultralytics/cfg/models/11/yolo11x.yaml')
    model = YOLO('ultralytics/cfg/models/ours/ours.yaml')
    #model = YOLO('ultralytics/cfg/models/ours/Ablation/optimized.yaml')
    model.train(data='ultralytics/cfg/datasets/CARPK.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=128,
                device='0, 1',
                optimizer='SGD', # using SGD
                patience=0, # close earlystop
                amp=True,
                # fraction=0.2,
                # project='runs/train/VisDrone/Ablation',
                project='runs/train/CARPK',
                name='GOOD-Net-l',
                )