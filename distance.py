import face_recognition

image = face_recognition.load_image_file("/home/mango/Downloads/group_img.jpg")
unknown_image = face_recognition.load_image_file("/home/mango/Downloads/nagrikta.jpeg")

biden_encoding = face_recognition.face_encodings(image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

spaces = face_recognition.face_distance(biden_encoding, unknown_encoding)
print(spaces)
