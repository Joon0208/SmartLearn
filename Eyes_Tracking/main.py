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

def positionEstimator(cropped_eye):
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

    if max_index == 0:
        eye_position = "RIGHT"
        color = [(0, 0, 0), (0, 255, 0)]
    elif max_index == 1:
        eye_position = 'CENTER'
        color = [(0, 255, 255), (255, 192, 203)]
    else:
        eye_position = 'LEFT'
        color = [(128, 128, 128), (0, 255, 255)]

    return eye_position, color

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
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

            eye_position_right, color_right = positionEstimator(crop_right)
            eye_position_left, color_left = positionEstimator(crop_left)

            cv.putText(frame, f'R: {eye_position_right}', (40, 220), FONTS, 1.0, color_right[1], 2, cv.LINE_AA)
            cv.putText(frame, f'L: {eye_position_left}', (40, 320), FONTS, 1.0, color_left[1], 2, cv.LINE_AA)

        end_time = time.time() - start_time
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
