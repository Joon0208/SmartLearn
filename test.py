import os
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Get the list of files in the "Faces" folder
faces_folder = "Templates/Faces/"
face_files = os.listdir(faces_folder)

# Check if the "Faces" folder is empty
if len(face_files) == 0:
    print("No data stored in the Faces folder.")
    register_face = input("Do you want to register a face? (y/n): ")

    if register_face.lower() == "y":
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Find all the faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("No face detected. Please try again.")
            else:
                # Take a picture of the first detected face
                x, y, w, h = faces[0]
                face_image = frame[y:y+h, x:x+w]

                # Generate a unique filename for the captured face
                num_files = len(os.listdir(faces_folder))
                filename = f"face_{num_files}.jpg"
                filepath = os.path.join(faces_folder, filename)

                # Save the captured face image
                cv2.imwrite(filepath, face_image)
                print(f"Face registered successfully. Image saved as {filename}")
                break

    else:
        print("Exiting...")
        exit()

else:
    # Load the face encodings and names from the files in the "Faces" folder
    known_face_encodings = []
    known_face_names = []
    for face_file in face_files:
        face_image = cv2.imread(os.path.join(faces_folder, face_file))
        face_encoding = None  # Placeholder for face encoding (not used in this code)
        face_name = os.path.splitext(face_file)[0]  # Extract the name from the file name

        known_face_encodings.append(face_encoding)
        known_face_names.append(face_name)

    # Initialize some variables
    face_locations = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert the image from BGR color to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if process_this_frame:
            face_locations = []
            face_names = []

            for (x, y, w, h) in faces:
                # Resize the face region of interest
                face_image = cv2.resize(rgb_frame[y:y+h, x:x+w], (150, 150))

                # Placeholder for face encoding (not used in this code)
                face_encoding = None

                face_locations.append((y, x+w, y+h, x))
                face_names.append("Unknown")

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
