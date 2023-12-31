from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, UserMixin, current_user, login_required
from Forms import *
from datetime import datetime

import base64, os
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math
import face_recognition
import io

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'hi'
socketio = SocketIO(app)

# Configure the database
DB_NAME = "database.db"
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
db = SQLAlchemy(app)

# Define the User class for database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150))
    date_joined = db.Column(db.Date, default=datetime.utcnow)
    role = db.Column(db.String(20))

# Define the Exam_Attempt class for database
class Exam_Attempt(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    non_center_eye_count = db.Column(db.Integer)
    time_taken = db.Column(db.String(20))
    score = db.Column(db.Integer)
    activity = db.Column(db.String(20))
    def __init__(self, name, non_center_eye_count, time_taken, score,activity):
        self.name = name
        self.non_center_eye_count = non_center_eye_count
        self.time_taken = time_taken
        self.score = score
        self.activity = activity

# Define the ImageModel class for database
class ImageModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(255), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'  # Update with your login route
login_manager.init_app(app)

# Load user data for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the home route
@app.route('/')
def home():
    if 'user' in session:
        user = session['user']
        user = user[1]
        return render_template('home.html', first_name=user)
    else:
        return render_template('home.html')

# Define the aboutUs route
@app.route('/aboutUs')
def about_us():
    return render_template('aboutUs.html')

# Define the signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get information from forms
        form = CreateUserForm(request.form)
        email = request.form.get('email')
        name = request.form.get('name')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        try:
            user = User.query.filter_by(email=email).first()

            if user:
                flash('Email already exists', category='error')
            elif len(email) < 4:
                flash('Invalid Email', category='error')
            elif len(name) < 2:
                flash('Invalid Name', category='error')
            elif password1 != password2:
                flash('Passwords do not match', category='error')

            else:
                new_user = User(email=email, name=name, password=password1, role='Student')
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user, remember=True)

                flash("Account created", category='success')
                return redirect(url_for('face_registration', name = name))  # Update with your login route

        except Exception as e:
            flash('An error occurred while creating the account', category='error')
            print(f"Error: {str(e)}")

    return render_template('signup.html', user=current_user)

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                session['user_id'] = user.id
                session['user_name'] = user.name
                session['role'] = user.role

                if user.role == 'Student':
                    return redirect(url_for('student_homepage'))
                elif user.role == 'Staff':
                    return redirect(url_for('staff_homepage'))
            else:
                flash('Incorrect password', category='error')
        else:
            flash('Email does not exist', category='error')

    return render_template('login.html', user=current_user)

# Define the logout route
@app.route('/logout')
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("role", None)

    print(session)

    return redirect(url_for('login'))

# Define the face_registration route
@app.route('/face_registration', methods=['GET','POST'])
def face_registration():
    if request.method == 'POST':
        user_name = request.args.get('name')
        if user_name is None:
            flash('User name not found', category='error')
            return redirect(url_for('signup'))

        face_image = request.form.get('face_image')
        face_image = face_image.split(',')[1].encode()
        image_data = base64.b64decode(face_image)

        # Create an instance of your ImageModel and set the values
        image_instance = ImageModel(user_name=user_name, image_data=image_data)

        # Add the instance to the session and commit the changes
        db.session.add(image_instance)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('face_registration.html', detection_status=False)

# Define the eye_tracking_video route
@app.route('/eye_tracking_video')
def eye_tracking_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the eye_tracking route
@app.route('/eye_tracking')
def eye_tracking():
    return render_template('eye_tracking.html')

# Define the student_homepage route
@app.route('/student_homepage')
@login_required
def student_homepage():
    user = User.query.get(current_user.id)  # Retrieve user information from the database

    return render_template('student_homepage.html', user=user)

@app.route('/student_profile')
@login_required
def student_profile():
    user = User.query.get(current_user.id)  # Retrieve user information from the database
    return render_template('student_profile.html', user=user)

# Define the staff_homepage route
@app.route('/staff_homepage')
@login_required
def staff_homepage():
    return render_template('staff_homepage.html')

# Define the exam route
@app.route('/exam')
@login_required
def exam():
    return render_template('exam.html')

# Define the exam1 route
@app.route('/exam1')
@login_required
def exam1():
    global stop_frame_generation
    stop_frame_generation = False

    return render_template('exam1.html')

# Define the exam_result
@app.route('/exam_result')
@login_required
def exam_result():
    # Query the Exam_Attempt model to get all attempts
    exam_attempts = Exam_Attempt.query.all()

    # Render the exam_result.html template with the exam_attempts data
    return render_template('exam_result.html', exam_attempts=exam_attempts)


# Facial Recognition Function

# Define the video_feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the face_lock route
@app.route('/face_lock')
@login_required
def face_lock():
    global fr

    if fr is None:
        fr = FaceRecognition()

    frame = None  # Capture a new frame here if needed
    detected_frame, detected_user = fr.run_recognition(frame)  # Pass the captured frame to run_recognition
    print(detected_user)

    current_user_name = current_user.name  # Assuming 'current_user' is a User object from Flask-Login

    allow_unlock = current_user_name == detected_user  # Check if detected user matches current user
    message = "User Authenticated. Click the button to proceed." if allow_unlock else "Wrong user detected. Please Refresh the page to authenticate again!"

    return render_template('face_lock.html', allow_unlock=allow_unlock, message=message, detected_user=detected_user, detected_frame=detected_frame)

fr = None  # FaceRecognition instance

# Define the gen_frames function for video streaming
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
            frame, detected_user = fr.run_recognition(frame)

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Define the face_confidence function for video streaming
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# Define the FaceRecognition class
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
        # Assuming you have a SQLAlchemy model named ImageModel
        with app.app_context():
            images = ImageModel.query.all()

        for image in images:
            face_image = face_recognition.load_image_file(io.BytesIO(image.image_data))
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.user_name)

        print(self.known_face_names)

    def run_recognition(self, frame=None):
        if frame is None:
            # Capture a frame from your camera or video feed
            success, frame = camera.read()  # Update this line as per your camera setup
            if not success:
                return None, None  # Return None if frame capture fails

        # Perform face recognition on the provided frame
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        detected_user = None
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

            name = name.split(".")[0]
            self.face_names.append(f'{name} ({confidence})')
            detected_user = name


        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Create the frame with the name
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
            cv.putText(frame, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        return frame, detected_user

# Exam Monitoring Function


# Initialize a flag to signal when to stop the frame generation
stop_frame_generation = False

# Route to handle the AJAX request to stop the camera
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global stop_frame_generation
    stop_frame_generation = True
    data = request.get_json()

    # Calculate the combined time string
    time_taken_minutes = int(elapsed_time_seconds // 60)
    time_taken_seconds = int(elapsed_time_seconds % 60)
    combined_time_string = f'{time_taken_minutes}min{time_taken_seconds}sec'

    # Determine activity based on non_center_eye_count
    activity = 'Suspicious' if non_center_eye_count > 200 else 'Not Suspicious'

    score = data.get('score', 0)


    # Create a new Exam_Attempt record and add it to the database
    exam_attempt = Exam_Attempt(
        name=session['user_name'],
        non_center_eye_count=non_center_eye_count,  # Replace with actual non_center_eye_count
        time_taken=combined_time_string,
        score=score,
        activity=activity)

    print(session['user_name'],non_center_eye_count,combined_time_string,score,activity)
    db.session.add(exam_attempt)
    db.session.commit()

    return jsonify({'message': 'Camera stopped successfully', 'time_taken': combined_time_string})



frame_counter = 0
FONTS = cv.FONT_HERSHEY_COMPLEX

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coords = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coords]
    return mesh_coords

def eyesExtractor(img, eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv.fillPoly(mask, [np.array(eye_coords, dtype=np.int32)], 255)
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155
    return eyes

def positionEstimator(cropped_eye):
    global non_center_eye_count  # Declare global variable to modify the count inside the function
    h, w = cropped_eye.shape
    gaussian_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussian_blur, 3)
    _, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece:piece+piece]
    left_piece = threshed_eye[0:h, piece+piece:w]

    eye_parts = [np.sum(right_piece == 0), np.sum(center_piece == 0), np.sum(left_piece == 0)]
    max_index = np.argmax(eye_parts)

    if max_index != 1:  # If eye is not in the center position
        non_center_eye_count += 1  # Increment the non-center eye count

    return eye_parts, max_index

# Initialize a flag to signal when to stop the frame generation
stop_frame_generation = False
# Initialize the non-center eye count
non_center_eye_count = 0
# Variable to store the time when the camera starts
start_time = 0
elapsed_time_seconds = 0

# Function to generate frames
def generate_frames():
    global stop_frame_generation
    global frame_counter
    global non_center_eye_count
    global start_time
    global elapsed_time_seconds

    # Get the current time to start counting
    start_time = time.time()

    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while not stop_frame_generation:
            frame_counter += 1

            # Check if the camera is off
            ret, frame = camera.read()
            if not ret:
                stop_frame_generation = True
                break

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)

                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right = eyesExtractor(frame, right_coords)
                crop_left = eyesExtractor(frame, left_coords)

                eye_parts_right, max_index_right = positionEstimator(crop_right)
                eye_parts_left, max_index_left = positionEstimator(crop_left)

                # cv.putText(frame, f'R: {max_index_right}', (40, 220), FONTS, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                # cv.putText(frame, f'L: {max_index_left}', (40, 320), FONTS, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'Non-Center Eye Count: {non_center_eye_count}', (40, 420), FONTS, 1.0, (255, 255, 255), 2, cv.LINE_AA)

                # Calculate the elapsed time and display it on the frame
                elapsed_time_seconds = time.time() - start_time
                elapsed_time_str = f'Time: {int(elapsed_time_seconds)}s'
                cv.putText(frame, elapsed_time_str, (40, 520), FONTS, 1.0, (255, 255, 255), 2, cv.LINE_AA)

            ret, buffer = cv.imencode('.jpg', frame)
            if not ret:
                continue

            # Emit the data through Flask-SocketIO
            socketio.emit('frame_data', {'elapsed_time': elapsed_time_seconds, 'non_center_eye_count': non_center_eye_count})

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera when loop ends
    camera.release()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the "users" table in the database
    app.run(debug=True)