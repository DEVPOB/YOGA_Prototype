import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
import pickle

app = Flask(__name__)

with open('model.p', 'rb') as file:
    model = pickle.load(file)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        data_aux = []
        success, frame = cap.read()
        if not success:
            break
        else:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    for body_landmark in results.pose_landmarks.landmark:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                        x = body_landmark.x
                        y = body_landmark.y
                        z = body_landmark.z

                        data_aux.append(x)
                        data_aux.append(y)
                    
                    if data_aux:
                        prediction = model.predict([np.asarray(data_aux)])
                        predictionSTR = str(prediction);
                        print(predictionSTR)
                    if predictionSTR == "['class26']" or predictionSTR == "['class21']":
                        print("Not Correct")
                    else:
                        print("Correct")
                except Exception as e:
                    print("Exception:", e)
                
                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('YOGA.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
