from flask import Flask, redirect, url_for, render_template, request, Response
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(-1) 
model = YOLO("yolov8n-face.pt")
# Real Time Face detection on webcam
def real_time_Faces():
    camera = cv2.VideoCapture(-1)  
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            cv2.imwrite("sample.jpg", frame)
            path = 'sample.jpg'
            result_frame = model.predict(source=path, show=False)
            boxes = result_frame[0].boxes
            boxIndx = []
            for box in boxes:
                boxIndx.append(box.data.tolist()[0])
            face_result_vid = np.array(boxIndx)

            for face_vid in face_result_vid:
                x1, y1, x2, y2, acc, _ = face_vid
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                imageVid = cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
                cv2.putText(imageVid, str(acc), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', imageVid)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    if not camera.release():
        camera.release()
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video')
def video():
    return Response(real_time_Faces(), mimetype='multipart/x-mixed-replace; boundary=frame')
