import cv2
import face_recognition
import os
import numpy as np

# Load known face images and names
known_face_images = "face_data"
known_persons_folder= "face_data"
known_face_names = "face_data/Elijah"

known_persons_encodings = {}
known_persons_encodings_array = {}


# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file(os.path.join("face_data", "Elijah.jpg"))
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file(os.path.join("face_data", "Justin.jpg"))
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Joe Biden",
    "Donald Trump"
]


# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    # print(known_persons_encodings)
    ret, frame = video_capture.read()
    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare current face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unkown"

         # Check if there is a match
        # if True in matches:
        #     matched_index = matches.index(True)
        #     name = known_face_names[matched_index]
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
        # Draw rectangle and label for the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()