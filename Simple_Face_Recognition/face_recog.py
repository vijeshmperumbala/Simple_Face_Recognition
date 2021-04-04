from imutils import paths
import face_recognition
import pickle
import cv2
import os

known_face = "Simple_Face_Recognition\\known_face"
known_face = os.path.join(known_face, 'mohanlal')
imagePaths = list(paths.list_images(known_face))
knownEncodings = []
knownNames = []
names = []
for i in imagePaths:
    for j in i:
        if j == "\\":
            newname = i.replace("\\",'/')
    names.append(newname)

for path in names:
    cls = path.split("/")[-2]
    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(cls)

data = {"encodings": knownEncodings, "names": knownNames}
f = open("Simple_Face_Recognition\\face_enc", "wb")
f.write(pickle.dumps(data))
f.close()

