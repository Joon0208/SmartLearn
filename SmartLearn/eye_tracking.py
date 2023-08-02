import cv2 as cv
import mediapipe as mp
import time
import numpy as np

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

# Initialize the non-center eye count
non_center_eye_count = 0

# Variable to store the time when the camera starts
start_time = 0

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

# Function to generate frames
def generate_frames():
    global stop_frame_generation
    global frame_counter
    global non_center_eye_count
    global start_time

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

# Now, when the camera starts (before calling generate_frames()), make sure to reset the flag
stop_frame_generation = False
