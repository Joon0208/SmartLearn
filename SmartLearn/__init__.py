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

from models import db
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy 

from os import path

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


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'this is a secret key'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
    app.config['MODELS_FOLDER'] = 'FYP/models'
    app.secret_key = 'this is a secret key'

    db.init_app(app)

    from models import User

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        with app.app_context():
            db.create_all()
            print('Database created')





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




if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the "users" table in the database
    app.run(debug=True)




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
    create_user_form = CreateUserForm(request.form)
    if request.method == 'POST' and create_user_form.validate():
        accounts_dict = {}
        db = shelve.open('storage.db', 'c')

        try:
            accounts_dict = db['Users']
        except:
            print("Error in retrieving Users from storage.db.")

        user = User.User(create_user_form.first_name.data, create_user_form.last_name.data, create_user_form.birthday.data, create_user_form.gender.data, create_user_form.email.data, create_user_form.phone_number.data, create_user_form.password.data)
        accounts_dict[user.get_user_id()] = user
        db['Users'] = accounts_dict
        db.close()
        # flash(f'Account Created Successfully for {create_user_form.first_name.data}', category='success')
        return redirect(url_for('face_registration'))
    return render_template('signup.html', form=create_user_form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    users_dict = {}
    db = shelve.open('storage.db','r')
    users_dict = db['Users']
    users_list = []

    # If there is no account in database, direct user to signup page
    if users_dict == {}:
        return redirect(url_for('signup'))

    else:
        for key in users_dict:
            user = users_dict.get(key)
            print(key)
            users_list.append(user)
            form = LogIn(request.form)
            if request.method == 'POST' and form.validate():
                for user in users_list:
                    if form.email.data == user.get_email() and form.password.data == user.get_password():
                        flash('Log in successfully!', 'success')

                        id = user.get_user_id()
                        first_name = user.get_first_name()
                        last_name = user.get_last_name()
                        birthday = user.get_birthday()
                        gender = user.get_gender()
                        email = user.get_email()
                        phone_number = user.get_phone_number()
                        password = user.get_password()

                        user_details = [id, first_name,last_name,birthday,gender,email,phone_number,password]
                        session['user'] = user_details

                        return redirect(url_for("student_homepage"))
                    else:
                        if "user" in session:
                            return render_template('login.html')
                        flash('Log in unsuccessful, please try again!','danger')
    return render_template('login.html', title='Login', form=form)


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
    session.pop("user", None)
    session.pop("staff", None)
    cart_dict = {}
    db = shelve.open('storage.db', 'c')
    cart_dict = db['Cart']
    cart_dict.clear()
    db['Cart'] = cart_dict
    db.close()

    return redirect(url_for('login'))


@app.route('/customer_update/<int:id>/', methods=['GET', 'POST'])
def customer_update(id):
    if 'user' in session:

        update_user_form = CreateUserForm(request.form)
        if request.method == 'POST' and update_user_form.validate():
            accounts_dict = {}
            db = shelve.open('storage.db', 'w')
            accounts_dict = db['Users']
            user = accounts_dict.get(id)
            user.set_first_name(update_user_form.first_name.data)
            user.set_last_name(update_user_form.last_name.data)
            user.set_birthday(update_user_form.birthday.data)
            user.set_gender(update_user_form.gender.data)
            user.set_email(update_user_form.email.data)
            user.set_phone_number(update_user_form.phone_number.data)
            user.set_password(update_user_form.password.data)
            # Update session's new list to show on customer's profile
            id = user.get_user_id()
            first_name = user.get_first_name()
            last_name = user.get_last_name()
            birthday = user.get_birthday()
            gender = user.get_gender()
            email = user.get_email()
            phone_number = user.get_phone_number()
            password = user.get_password()
            session['user'] = [id,first_name,last_name,birthday,gender,email,phone_number,password]
            # Update new details to the database
            db['Users'] = accounts_dict
            db.close()
            return redirect(url_for('customer_profile'))
        else:
            accounts_dict = {}
            db = shelve.open('storage.db', 'r')
            accounts_dict = db['Users']
            db.close()
            user = accounts_dict.get(id)
            update_user_form.first_name.data = user.get_first_name()
            update_user_form.last_name.data = user.get_last_name()
            update_user_form.birthday.data = user.get_birthday()
            update_user_form.gender.data = user.get_gender()
            update_user_form.email.data = user.get_email()
            update_user_form.phone_number.data = user.get_phone_number()
            update_user_form.password.data = user.get_password()
        return render_template('account.html', form=update_user_form)
    else:
        return redirect(url_for('login'))

@app.route('/updateUser/<int:id>/', methods=['GET', 'POST'])
def update_user(id):
    update_user_form = CreateUserForm(request.form)
    if request.method == 'POST' and update_user_form.validate():
        accounts_dict = {}
        db = shelve.open('storage.db', 'w')
        accounts_dict = db['Users']

        user = accounts_dict.get(id)
        user.set_first_name(update_user_form.first_name.data)
        user.set_last_name(update_user_form.last_name.data)
        user.set_birthday(update_user_form.birthday.data)
        user.set_gender(update_user_form.gender.data)
        user.set_email(update_user_form.email.data)
        user.set_phone_number(update_user_form.phone_number.data)
        user.set_password(update_user_form.password.data)

        db['Users'] = accounts_dict
        db.close()

        return redirect(url_for('accounts'))
    else:
        accounts_dict = {}
        db = shelve.open('storage.db', 'r')
        accounts_dict = db['Users']
        db.close()

        user = accounts_dict.get(id)
        update_user_form.first_name.data = user.get_first_name()
        update_user_form.last_name.data = user.get_last_name()
        update_user_form.birthday.data = user.get_birthday()
        update_user_form.gender.data = user.get_gender()
        update_user_form.email.data = user.get_email()
        update_user_form.phone_number.data = user.get_phone_number()
        update_user_form.password.data = user.get_password()

    return render_template('account.html', form=update_user_form)

@app.route('/deleteUser/<int:id>', methods=['POST'])
def delete_user(id):
    users_dict = {}
    db = shelve.open('storage.db', 'w')
    users_dict = db['Users']

    users_dict.pop(id)

    db['Users'] = users_dict
    db.close()

    if 'user' in session:
        return redirect(url_for('logout'))
    elif 'staff' in session:
        return redirect(url_for('accounts'))

    return redirect(url_for('logout'))

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


    # Create a new Users record and add it to the database
    user = User(name="John Doe", non_center_eye_count=non_center_eye_count, elapsed_time_seconds=elapsed_time_seconds)
    db.session.add(user)
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


