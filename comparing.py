import face_recognition


known_image = face_recognition.load_image_file("/home/mango/Downloads/group_img.jpg")
another_known_image = face_recognition.load_image_file("/home/mango/Downloads/download.jpeg")
unknown_image = face_recognition.load_image_file("/home/mango/Downloads/nagrikta.jpeg")
trying = face_recognition.face_landmarks(known_image)
known_image_encoding1 = face_recognition.face_encodings(known_image)[0]
known_image_encoding2 = face_recognition.face_encodings(another_known_image)[0]
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([known_image_encoding1, known_image_encoding2], unknown_image_encoding)

dist = face_recognition.face_distance([known_image_encoding1, known_image_encoding2], unknown_image_encoding)
print(dist)
print(1-dist[1])
