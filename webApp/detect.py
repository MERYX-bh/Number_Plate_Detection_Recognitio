import ultralytics
from ultralytics import YOLO
from matplotlib import pyplot as plt

import cv2
import easyocr

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

reader = easyocr.Reader(['en'], gpu= True)

model = YOLO('yolov8n.pt')  # load an official model
model = YOLO("best_plate.pt")

def ocr_image(img, coordinates):
  x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
  # we will crop the image to keep only the plate detected
  img = img[y:h , x:w]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  result = reader.readtext(gray)

  text = ""
  
  for res in result:
    if len(result) == 1:
      text = res[1] 
    
    elif len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
      text = res[1]
  
  return text

def annotate_image(path):
    img = cv2.imread(path)
    results = model.predict(img)
    for result in results:
        xyxy = result.boxes.xyxy
        xyxy = xyxy.tolist()
        text_ocr = ocr_image(img, xyxy[0])
        label = text_ocr

        coordinates = result.boxes.xyxy[0]
        x, y = int(coordinates[0]), int(coordinates[1])
        coordinates = result.boxes.xywh[0]
        w, h = int(coordinates[2]), int(coordinates[3])

        print("x",x)
        print("y",y)
        print("w",w)
        print("h",h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Write label and confidence on the image
        cv2.rectangle(img, (x, y-30), (x + len(label) * 17, y - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{label}", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


    return img, label
