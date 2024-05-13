from ultralytics import YOLO
from tqdm import tqdm

# Load a model
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO("/home/yuanzf/uvi-time-machine/_script/A-city-never-was/runs/classify/train4/weights/best.pt")
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"

# Train the model
results = model.train(data=YOLOFOLDER, epochs=50, imgsz=416, workers = 4, resume=True)
