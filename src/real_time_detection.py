import asyncio
import time

from ultralytics import YOLO
import cv2
import easyocr

from utils.util import license_complies_format
from utils.fetch_vehicle_data import fetch_vehicle_data

reader = easyocr.Reader(['en'], gpu=False)


model = YOLO('./models/license_plate_detector.pt', task='detect')

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../outputs.avi', fourcc, 10, (frame_width, frame_height))

print('process starts')
ret = True
while ret:
    tic = time.time()
    ret, frame = cap.read()
    if ret:
        # Perform inference on the frame
        output = model(frame, imgsz=320, verbose=False)

        for i, det in enumerate(output[0].boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = det

            if score < 0.6:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop the license plate from the frame
            license_plate_crop = frame[y1:y2, x1:x2]

            # Read the license plate text if confidence > 0.70
            if score >= 0.7:
                # Perform OCR on the cropped license plate region
                detections = reader.readtext(license_plate_crop)
                for detection in detections:
                    bbox, text, ocr_score = detection
                    if ocr_score >= 0.7:
                        # Print the license plate text
                        if license_complies_format(text):
                            license_plate = text.replace(" ", "").replace("-", "").replace("'","").upper()

                            car_details = asyncio.run(fetch_vehicle_data(license_plate))
                            if car_details["error"] == False:
                                print(f"License Plate: {car_details['license_plate']}, the car's model is : {car_details['vehicle_brand'], car_details['vehicle_model']}")
                        else:
                            print(f"{text} did not comply")       
            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, 'license_plate', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Display FPS
        cv2.putText(frame, 'FPS: ' + str(int(1 / (time.time() - tic))), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()