import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from src.utils.util import read_license_plate
from src.utils.fetch_vehicle_data import fetch_vehicle_data

from fastapi import FastAPI, UploadFile, File
app = FastAPI()

license_plate_detector = YOLO('./models/license_plate_detector.pt')

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the image from the post request
        image_bytes = await file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Detect any license plates
        license_plates = license_plate_detector(frame)[0]
        
        # Iterate through the detected license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Ensure valid bounding box before cropping
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            h, w, _ = frame.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            license_plate_crop = frame[y1:y2, x1:x2, :]
            hsv_image = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([15, 100, 100])  # Lower bound for darker yellow
            upper_yellow = np.array([40, 255, 255])  # Upper bound for yellow
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            license_plate_crop_thresh = cv2.bitwise_and(license_plate_crop, license_plate_crop, mask=yellow_mask)
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text:
                vehicle_data = await fetch_vehicle_data(license_plate_text)
                return {"license_plate": license_plate_text, "vehicle_data": vehicle_data}
        
        # If no license plate is found
        return {"message": "Couldn't find data"}
    
    except Exception as e:
        return {"error": str(e)}



# To run the app, use the following command:
# uvicorn app:app --reload --host 0.0.0.0 --port 7020
