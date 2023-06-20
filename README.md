<H1 align="center">Automatic Number Plate Detection and Recognition using YOLOv8</H1>

**Project description**

This mainly a license plate recognition based on deep learning.

- First we start by detecting the license plate then when detected, we make the characters recognitions.

plate detection : YOLOV8 

Characters recognition : EasyOCR

- Finally I deployed the model on a web app using Flask.


We start by detecting the license plate using yolov8 (you can find the detailed code in the notebook ) of the car like in the video below:

[![Video Preview](https://github.com/MERYX-bh/Car-plate-recognition/blob/main/preview_plate.png)](https://github.com/MERYX-bh/Car-plate-recognition/blob/main/t%C3%A9l%C3%A9chargement.mp4)

Then we continue by recognizing the characters on the plate using easy ocr see the video below:

[![Video Preview](https://github.com/MERYX-bh/Car-plate-recognition/blob/main/detection_on_image.jpg)](https://github.com/MERYX-bh/Car-plate-recognition/blob/main/ocr.mp4)

Finally I deploed the final model on a web app using Flask:

Here's a demo of the app [Demo video](https://github.com/MERYX-bh/Car-plate-recognition/blob/main/Screen_recording.mp4)


