import cv2
import tensorflow as tf

categories = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']


def prepare(path):
    img_size = 48
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.load_model('model_filter.h5')
prediction1 = model.predict([prepare('1.jpg')])
print(prediction1)
