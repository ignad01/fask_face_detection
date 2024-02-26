from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

import numpy as np
from PIL import Image
import os

model = YOLO("yolov8n-face.pt")
# img_path = "classPic.jpg"
# croped_image_path = r"C:\Users\piyus\OneDrive\Documents\MTechProject\YOLO\face-detection-yolov8\cropFaces"
# results_img = model.predict(source=img_path, show=False)

# face_result = np.array(results_img[0])

# img1 = cv2.imread(img_path)
# for ind, face in enumerate(face_result):
#     x1, y1, x2, y2, acc, _ = face
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     if acc > 0.6:
#         faceImg = img1[y1:y2, x1:x2]
#         faceImg = cv2.resize(faceImg, (250, 250))
#         cv2.imwrite(os.path.join(croped_image_path,str(ind)+".jpg"), faceImg)
#     image = cv2.rectangle(img1, (x1, y1), (x2,y2), (0, 255, 0), 1)
# #     cv2.putText(image, str(acc), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
# cv2.imwrite("box.jpg", image)
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break
    else:
        cv2.imwrite("sample.jpg", frame)
        path = 'sample.jpg'
        result_frame = model.predict(source=path, show=False)
        face_result_vid = np.array(result_frame[0])
        for face_vid in face_result_vid:
            x1, y1, x2, y2, acc, _ = face_vid
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            imageVid = cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
            cv2.putText(imageVid, str(acc), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.resize(imageVid, (640, 480))
    cv2.imshow("frame", imageVid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyWindow("frame")
