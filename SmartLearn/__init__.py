from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from Forms import *
import shelve, User, base64, os
from eye_tracking import generate_frames
from facial_recognition import gen_frames
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, UserMixin, current_user
from datetime import datetime


#Login mail@mail.com
#password 12345

#Login2 nomail@mail.com
#password2 54321


app = Flask(__name__)
app.secret_key = 'hi'
socketio = SocketIO(app)

DB_NAME = "database.db"
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'  # Update with your login route
login_manager.init_app(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150))
    date_joined = db.Column(db.Date, default=datetime.utcnow)
    role = db.Column(db.String(20))

from flask_login import UserMixin

class Exam_Attempt(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    non_center_eye_count = db.Column(db.Integer)
    time_taken = db.Column(db.Float)

    def __init__(self, name, non_center_eye_count, time_taken):
        self.name = name
        self.non_center_eye_count = non_center_eye_count
        self.time_taken = time_taken


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the Users model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     non_center_eye_count = db.Column(db.Integer, nullable=False)
#     elapsed_time_seconds = db.Column(db.Float, nullable=False)

#     def __init__(self, name, non_center_eye_count, elapsed_time_seconds):
#         self.name = name
#         self.non_center_eye_count = non_center_eye_count
#         self.elapsed_time_seconds = elapsed_time_seconds

# @app.route('/')
# def home():
#     return render_template('home.html')

@app.route('/')
def home():
    if 'user' in session:
        user = session['user']
        user = user[1]
        return render_template('home.html', first_name=user)
    else:
        return render_template('home.html')

@app.route('/aboutUs')
def about_us():
    return render_template('aboutUs.html')

@app.route('/returnHome')
def return_home():
    return render_template('returnHome.html')

# JiaJun
# Staff pages
@app.route('/stafflogin', methods=['GET', 'POST'])
def stafflogin():

    form = LogIn(request.form)
    if request.method == 'POST' and form.validate():
        if form.email.data == "staff@account" and form.password.data == 'staffpass':
            flash('Log in successfully!', 'success')
            session['staff'] = 'staff'
            return redirect(url_for("staff_homepage"))
        else:
            flash('Log in unsuccessful, please try again!','danger')
    return render_template('stafflogin.html', title='Login', form=form)

@app.route('/accounts')
def accounts():
    accounts_dict = {}
    db = shelve.open('storage.db', 'r')
    accounts_dict = db['Users']
    db.close()
    accounts_list = []
    for key in accounts_dict:
        user = accounts_dict.get(key)
        accounts_list.append(user)

    return render_template('accounts.html', count=len(accounts_list), users_list=accounts_list)

#  Customer pages
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
                # add user to database
                print(email)
                print(name)
                new_user = User(email=email, name=name, password=password1, role='Student')
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user, remember=True)

                flash("Account created", category='success')
                return redirect(url_for('login'))  # Update with your login route

        except Exception as e:
            flash('An error occurred while creating the account', category='error')
            print(f"Error: {str(e)}")

    return render_template('signup.html', user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    # When you access request inside of a route, it will have information about the request that was sent to access this route
    # It will say the URL, the method
    # We can access the form attribute of our request; has all of the data that was sent as a part of the form
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if (user.password, password):
                flash('Logged in successfullly!', category='success')
                login_user(user, remember=True)
                session['user_id'] = user.id
                session['user_name'] = user.name
                session['role'] = user.role

                return redirect(url_for('student_homepage'))
            else:
                flash('Incorrect password', category='error')
        else:
            flash('Email does not exist', category='error')


    # login_url = url_for('auth.callback', _external=True)
    return render_template('login.html', user=current_user)



# Customer have to log out after logging in their account
# One disadvantage is customer have to logout of their account to see their edits
@app.route('/customer_profile', methods = ['GET', 'POST'])
def customer_profile():
    if 'user' in session:
        user = session["user"]
        accounts_list = user[1:8]
        id = user[0]
        return render_template('customer_profile.html', accounts_list=accounts_list, id=id)
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("role", None)

    print(session)

    return redirect(url_for('login'))


@app.route('/face_registration', methods=['GET', 'POST'])
def face_registration():
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
        return render_template('face_registration.html', detection_status=True, face_image=image_path,
                               success_message='Image saved successfully.')

    return render_template('face_registration.html', detection_status=False)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eye_tracking_video')
def eye_tracking_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eye_tracking')
def eye_tracking():
    return render_template('eye_tracking.html')

@app.route('/exam_monitoring')
def exam_monitoring():
    return render_template('exam_monitoring.html')

@app.route('/student_homepage')
def student_homepage():
    return render_template('student_homepage.html')

@app.route('/staff_homepage')
def staff_homepage():
    return render_template('staff_homepage.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')

@app.route('/exam1')
def exam1():
    global stop_frame_generation
    stop_frame_generation = False

    return render_template('exam1.html')

@app.route('/exam_result')
def exam_result():
    # Query the Exam_Attempt model to get all attempts
    exam_attempts = Exam_Attempt.query.all()

    # Render the exam_result.html template with the exam_attempts data
    return render_template('exam_result.html', exam_attempts=exam_attempts)


# Initialize a flag to signal when to stop the frame generation
stop_frame_generation = False

# @app.route('/submit_score', methods=['POST'])
# def submit_score():
#     global score
#     data = request.get_json()
#     score = data.get('score')
#
#     # Do whatever you want to do with the score here
#     print(f'Score submitted: {score}')
#     return jsonify({'message': 'Score submitted successfully', 'score': int(score)})

# Route to handle the AJAX request to stop the camera
@app.route('/stop_camera')
def stop_camera():
    global stop_frame_generation
    # global score
    stop_frame_generation = True
    print(non_center_eye_count)
    print(int(elapsed_time_seconds))
    # score = submit_score.__get__(score)


    # Create a new Exam record and add it to the database
    exam_attempts = Exam_Attempt(name=session['user_name'], non_center_eye_count=non_center_eye_count, time_taken=elapsed_time_seconds)
    db.session.add(exam_attempts)
    print("Non-center_eye_count and Time added to the database")
    db.session.commit()

    return jsonify({'message': 'Camera stopped successfully'})

# eye_tracking function

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