# Switch-Tray-Verification
from ultralytics import YOLO
# Python code to read image
import cv2

# To read image from disk, we use
# cv2.imread function, in below method,
#img = cv2.imread("C:/Users/shikh/Downloads/stv.jpg", cv2.IMREAD_COLOR)

# Load a model
#model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("C:/Users/shikh/Downloads/best (3)/best.pt")  # load a custom model

# Predict with the model
results = model("C:/Users/shikh/Downloads/stv.jpg")  # pC:/Users/shikh/Downloads/stv.jpg) predict on an image
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
