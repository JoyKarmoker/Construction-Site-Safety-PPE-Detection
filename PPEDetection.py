from ultralytics import YOLO  # Import YOLO model for object detection
import cv2  # Import OpenCV library for image processing
import cvzone  # Import cvzone library for drawing bounding boxes and text
import math  # Import math library for mathematical operations
import os  # Import os library for file and directory operations

# Load YOLO model with the specified weights file
model = YOLO("ppe.pt")

# Open the video file for processing
cap = cv2.VideoCapture("../Videos/ppe-1.mp4")

# Create a folder to store the output frames if it doesn't exist
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Define the class names for the objects that can be detected by the YOLO model
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask',
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus',
              'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

# Initialize the color for bounding boxes and text
myColor = (0, 0, 255)  # Default color is red

# Variables for calculating frames per second (FPS)
prev_frame_time = 0
new_frame_time = 0

# Main loop for processing each frame of the video
while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break  # Exit the loop if there are no more frames to read

    # Perform object detection with YOLO on the current frame
    results = model(img, stream=True)

    # Process the results of object detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Extract confidence and class label
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Update color based on class label
            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    myColor = (0, 0, 255)  # Red for non-compliance
                elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    myColor = (0, 255, 0)  # Green for compliance
                else:
                    myColor = (255, 0, 0)  # Blue for other classes

                # Draw text with class label and confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                   colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # Save the frame with YOLO detections applied
    output_path = os.path.join(output_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg")
    cv2.imwrite(output_path, img)

    # Display the processed frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release the video capture
cap.release()

# Recreate a video from the saved frames
output_video_path = "output_video.mp4"
img_array = []
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Video created successfully.")
