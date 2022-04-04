import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import tqdm

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
size_of_image = 150
folderPath = "test/test_sample"
X_pred = []

for i in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, i))
        img = cv2.resize(img, (size_of_image, size_of_image))
        X_pred.append(img)

X_pred = np.array(X_pred)

reconstructed_model = keras.models.load_model('Tumor_Prediction_Model.h5')
reconstructed_model.summary()

image_prediction = reconstructed_model.predict(X_pred)
print(image_prediction)