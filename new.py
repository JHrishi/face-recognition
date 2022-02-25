# we will use two famous Indian cricketers Ms Dhoni and V. Kohli for our projects
import face_recognition
import cv2
img = cv2.imread('one.jpg')

# Load a sample picture and learn how to recognize it.
ms_image = face_recognition.load_image_file("ms.jpg")
ms_face_encoding = face_recognition.face_encodings(ms_image)[0]

# Load a second sample picture and learn how to recognize it.
vk_image = face_recognition.load_image_file("virat-kohli.jpg")
vk_face_encoding = face_recognition.face_encodings(vk_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ms_face_encoding,
    vk_face_encoding
]
known_face_names = [
    "M.S Dhoni",
    "Virat Kohli"
]

face_names = []

# Find all the faces and face encodings in the image
face_locations = face_recognition.face_locations(img)
face_encodings = face_recognition.face_encodings(img, face_locations)

for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Draw a box around the face
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    # Draw a label with a name below the face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, name, (left - 10, top - 10), font, 1.0, (255, 255, 255), 1)

# Display the resulting image
cv2.imshow('recognized', img)
cv2.waitKey(0)
