import cv2
import numpy as np
from plyer import notification
import threading
from joblib import load
import mediapipe as mp
from skimage.feature import hog
import time
import json

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load posture detection model
posture_model = load('posture_detector_model.pkl')

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    return resized

# Function to send notification for bad posture
def notify():
    notification.notify(
        title='Posture Notification',
        message='Your current posture is BAD, Please Fix!',
        app_icon=None,
        timeout=10
    )

# Function to reset the notification flag after a timeout
def reset_notification_flag():
    global send_notification
    send_notification = False

# Initialize notification flag and time tracker
send_notification = False
bad_posture_start_time = None
bad_posture_duration_threshold = 5  # seconds

# Log file path
log_file_path = 'posture_log.json'

# Initialize posture log
posture_log = []

# Function to log posture data
def log_posture(posture_value):
    global posture_log
    timestamp = time.time()
    posture_log.append({'timestamp': timestamp, 'posture': posture_value})
    # Save the log to a file
    with open(log_file_path, 'w') as log_file:
        json.dump(posture_log, log_file)

# Function to start posture detection
def startPostureDetection():
    global send_notification, bad_posture_start_time

    # Initialize camera
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame (stream end?). Exiting ...")
                break

            # Preprocess frame
            preprocessed_frame = preprocess_image(frame)

            # Extract HOG features
            fd = hog(preprocessed_frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)

            # Predict posture
            prediction = posture_model.predict(fd.reshape(1, -1))
            posture_value = "bad" if prediction == 0 else "good"

            # Log posture data
            log_posture(posture_value)

            # Set font and text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 0, 255) if posture_value == "bad" else (0, 255, 0)

            # Check for bad posture and send notification if detected
            current_time = time.time()
            if posture_value == 'bad':
                if bad_posture_start_time is None:
                    bad_posture_start_time = current_time
                elif (current_time - bad_posture_start_time >= bad_posture_duration_threshold) and not send_notification:
                    notify()
                    send_notification = True
                    threading.Timer(bad_posture_duration_threshold, reset_notification_flag).start()
            else:
                bad_posture_start_time = None

            # Overlay the predicted posture text on the frame
            cv2.putText(frame, f"Posture: {posture_value}", (10, 30), font, font_scale, font_color, 2)

            # Convert the frame to RGB for landmark detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect landmarks
            results = pose.process(rgb_frame)

            # Draw Pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

            # Display the frame with the predicted posture and landmarks
            cv2.imshow('Posture Detection', frame)

            # Check for user input to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Start posture detection
startPostureDetection()
