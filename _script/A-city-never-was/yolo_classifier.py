from ultralytics import YOLO
from tqdm import tqdm

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
YOLOFOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_transformed/t_classifier_img_yolo8"

# Train the model
results = model.train(data=YOLOFOLDER, epochs=10, imgsz=416)