from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from Forms import *
import shelve, User, numpy as np, math, cv2, base64, os, face_recognition

#Login mail@mail.com
#password 12345

#Login2 nomail@mail.com
#password2 54321

app = Flask(__name__)
app.secret_key = 'hi'


camera = cv2.VideoCapture(0)
fr = None  # FaceRecognition instance

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

    return render_template('login.html', detection_status=False)

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
    return render_template('exam1.html')

if __name__ == '__main__':
    app.run()

