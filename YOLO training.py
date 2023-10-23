from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
    data='data.yaml',
    imgsz=416,  # smaller image size for faster training
    epochs=30,  # increased epochs for better training
    batch=8,
    name='end_model'
)

metrics = model.val()

# Save trained weights
model.save_weights('end_model.pt')







#https://medium.com/dkatalis/training-a-custom-object-detector-in-half-a-day-with-yolov8-5e1475fe201e