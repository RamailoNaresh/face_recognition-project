import face_recognition

image = face_recognition.load_image_file('/home/mango/Downloads/naresh.jpg')
locate = face_recognition.face_locations(image)
print(locate)
