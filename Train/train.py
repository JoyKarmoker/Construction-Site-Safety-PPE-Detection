from ultralytics import YOLO
model = YOLO('../../Yolo-Weights/yolov8l.pt')

# # Predict with the model
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk


# Train the model
results = model.train(data='Construction_Site_Safety/data.yaml', epochs=1, imgsz=640, batch=8)

