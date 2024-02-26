from flask import Flask, redirect, url_for, render_template, request, Response
import cv2
from ultralytics import YOLO
import numpy as np
#from PIL import Image
#import os

app = Flask(__name__)

#camera = cv2.VideoCapture(0)

model = YOLO("yolov8n-face.pt")
## Model part ###
# vid_path = "demo1.mp4"
# results_video = model.predict(source=vid_path, show=False)
# camera = cv2.VideoCapture(vid_path)
################

# def detect_faces():
#     for imgVid in results_video:
#         success, frame = camera.read()
        
#         if not success:
#             break
#         else:
#             face_result_vid = np.array(imgVid)

#             for face_vid in face_result_vid:
#                 x1, y1, x2, y2, acc, _ = face_vid
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 imageVid = cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
#                 cv2.putText(imageVid, str(acc), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#         yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Real Time Face detection on webcam
camera = cv2.VideoCapture(0)
def real_time_Faces():  
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

            ret, buffer = cv2.imencode('.jpg', imageVid)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
  

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)