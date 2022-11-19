import os
import cv2
import numpy as np
import face_recognition

frame_sizing = 0.5
path = 'images'
images = []
class_names = []

for cl in os.listdir(path):
    images.append(cv2.imread(f'{path}/{cl}'))
    class_names.append(os.path.splitext(cl)[0])

def encoding_images(images):
    encoding_list = []
    for image in images:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_encoding = face_recognition.face_encodings(rgb_image)[0]
        encoding_list.append(image_encoding)
    print('Encoding images loaded ✅')
    return encoding_list

print('⭐ Made with ❤️ by Duc + Hung ⭐')
print(class_names)
print('Encoding images loading ⚡')
encoding_images_loaded = encoding_images(images)

cap = cv2.VideoCapture(0)
print('Started camera 🚀')

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (0, 0), None, fx=frame_sizing, fy=frame_sizing)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encoding_images_loaded, face_encoding)
        face_distances = face_recognition.face_distance(encoding_images_loaded, face_encoding)
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < 0.40:
            name = class_names[best_match_index].upper() + f' {str(int(100 - round(face_distances[best_match_index], 2) * 100))}%'
        else:
            name = 'Unknown'

        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = int(y1 / frame_sizing), int(x2 / frame_sizing), int(y2 / frame_sizing), int(x1 / frame_sizing)
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    
    cv2.imshow('Face Recognition', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
