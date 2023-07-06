from flask import Flask, render_template, request, flash, Response
import face_recognition
import cv2
import os
import numpy as np
import base64
import time
import math

app = Flask(__name__)

known_faces = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/student_experience')
def student_experience():
    return render_template('student_experience.html')

@app.route('/exam_monitor')
def exam_monitor():
    return render_template('exam_monitor.html')


@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'POST':
        name = request.form.get('name')
        face_image = request.form.get('face_image')

        # Convert the base64 image data to bytes
        face_image = face_image.split(',')[1].encode()
        # Decode the image and save it to a file
        image_data = base64.b64decode(face_image)
        # Generate a unique filename using the current timestamp
        image_filename = f'{name}.jpg'
        # Create a directory for the user if it doesn't exist
        user_dir = os.path.join('faces', image_filename)
        # Save the captured image under the user's directory
        image_path = os.path.join(user_dir)
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # Render the template with the success message
        return render_template('register_face.html', detection_status=True, face_image=image_path,
                               success_message='Image saved successfully.')

    return render_template('register_face.html', detection_status=False)


camera = cv2.VideoCapture(0)
fr = None  # FaceRecognition instance

def gen_frames():
    global fr
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Perform face recognition on the frame
            if fr is None:
                fr = FaceRecognition()
            frame = fr.run_recognition(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self, frame):
        # Perform face recognition on the provided frame
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '???'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append(f'{name} ({confidence})')

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Create the frame with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame

if __name__ == '__main__':
    app.secret_key = 'secret_key'
    app.run(debug=True)