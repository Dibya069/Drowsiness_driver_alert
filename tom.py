# Import libraries
import cv2
import mediapipe as mp
import numpy as np

# Define constants
EAR_THRESHOLD = 0.2 # The threshold for detecting drowsiness
EAR_CONSEC_FRAMES = 10 # The number of consecutive frames with low EAR to trigger an alert
COUNTER = 0 # The counter for consecutive frames
ALARM_ON = False # The flag for alarm status

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Define a function to compute the euclidean distance between two points
def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Define a function to compute the eye aspect ratio
def eye_aspect_ratio(eye):
    # Compute the distances between the eye landmarks
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    c = euclidean_dist(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)
    return ear

# Define a function to get the eye landmarks from the face mesh
def get_eye_landmarks(face_landmarks, side):
    # Get the eye landmarks based on the side (left or right)
    if side == "left":
        eye_landmarks = [face_landmarks[33], face_landmarks[246], face_landmarks[161], face_landmarks[160], face_landmarks[159], face_landmarks[158]]
    elif side == "right":
        eye_landmarks = [face_landmarks[263], face_landmarks[466], face_landmarks[388], face_landmarks[387], face_landmarks[386], face_landmarks[385]]
    else:
        raise ValueError("Invalid side")
    # Convert the landmarks to a list of tuples
    eye_landmarks = [(int(landmark.x * image_width), int(landmark.y * image_height)) for landmark in eye_landmarks]
    return eye_landmarks

# Start the video capture
cap = cv2.VideoCapture(0)

# Loop over the frames
while cap.isOpened():
    # Read the frame
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    # Get the image dimensions
    image_height, image_width, _ = image.shape

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and find the face landmarks
    results = face_mesh.process(image)
    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If there are face landmarks, draw them and compute the EAR
    if results.multi_face_landmarks:
        # Loop over each face
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            # Get the left and right eye landmarks
            left_eye = get_eye_landmarks(face_landmarks, "left")
            right_eye = get_eye_landmarks(face_landmarks, "right")
            # Compute the EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            # Compute the average EAR
            ear = (left_ear + right_ear) / 2.0
            # Draw the EAR on the image
            cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Check if the EAR is below the threshold
            if ear < EAR_THRESHOLD:
                # Increment the counter
                COUNTER += 1
                # If the counter exceeds the limit, sound the alarm
                if COUNTER >= EAR_CONSEC_FRAMES:
                    # Set the alarm flag
                    ALARM_ON = True
                    # Draw an alert on the image
                    cv2.putText(image, "DROWSINESS ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Reset the counter and the alarm
                COUNTER = 0
                ALARM_ON = False

    # Display the image
    cv2.imshow('Drowsiness Detection', image)
    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
