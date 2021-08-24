import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# loading learned model
model = load_model('./trained_model/checkpoints')

# Utils function
def prediction_map(pred):
    idx = pred.argmax()
    return 'Without Mask' if idx == 1 else 'With Mask'


cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Converting img into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = img[x: x+w, y:y+h]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Model takes image with (128, 128), resize image
        resized_img = cv2.resize(face, (128, 128))[np.newaxis, ..., np.newaxis]

        # Prediction
        face_gen = ImageDataGenerator(rescale=1/255.0).flow(resized_img)
        prediction = model.predict(face_gen)

        # Converting predtion into class
        text = prediction_map(prediction)

        # Putting prediction text on the image
        cv2.putText(img, text, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('Video Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
