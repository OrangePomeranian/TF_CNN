import os
import io
from warnings import filterwarnings
import numpy as np
from numpy import expand_dims
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from time import perf_counter

colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']
colors_red = ['#331313', '#582626', '#9E1717', '#D35151', '#E9B4B4']
colors_dark = ['#1F1F1F', '#313131', '#636363', '#AEAEAE', '#DADADA']
sns.palplot(colors_green)
sns.palplot(colors_red)
sns.palplot(colors_dark)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

global iterations_on_data
iterations_on_data = 12
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
size_of_image = 150
X_train = []
y_train = []

for i in labels:
    folderPath = os.path.join("kaggle/input/brain-tumor-classification-mri", "Training", i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img,(size_of_image, size_of_image))

        if i == "no_tumor":
            data = img_to_array(img)
            samples = expand_dims(data, 0)
            datagen = ImageDataGenerator(rotation_range = 90, horizontal_flip = True)
            it = datagen.flow(samples, batch_size = 1)

            for k in range(2):
                batch = it.next()
                image = batch[0].astype('uint8')
                X_train.append(image)
                y_train.append(i)

        else:
            X_train.append(img)
            y_train.append(i)

for i in labels:
    folderPath = os.path.join("kaggle/input/brain-tumor-classification-mri", "Testing", i)

    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (size_of_image, size_of_image))

        X_train.append(img)
        y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 101 )

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))

y_train = y_train_new
y_test = y_test_new

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

effnet_premodel = EfficientNetV2B0(
    include_top = False,
    weights = 'imagenet',
    input_tensor = None,
    input_shape = (size_of_image, size_of_image, 3),
    classifier_activation = 'softmax',
    include_preprocessing = True)

model = effnet_premodel.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate = 0.5)(model)
model = tf.keras.layers.Dense(4, activation = 'softmax')(model)
model = tf.keras.models.Model(inputs = effnet_premodel.input, outputs = model)

print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

tensorboard = TensorBoard(log_dir = 'log')
checkpoint = ModelCheckpoint("effnet.h5",
                             monitor = "val_accuracy",
                             save_best_only = True,
                             mode = "auto",
                             verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor = "val_accuracy",
                              factor = 0.3,
                              patience = 2,
                              min_delta = 0.001,
                              mode = "auto",
                              verbose = 1)
t_start = perf_counter()

history = model.fit(X_train, y_train,
                    validation_split = 0.1,
                    epochs = iterations_on_data,
                    verbose = 1, batch_size = 32,
                    callbacks = [tensorboard, checkpoint, reduce_lr])

#zapis modelu
model.save("Tumor_Prediction_Model.h5")

filterwarnings('ignore')
t_stop = perf_counter()

epochs = [i for i in range(iterations_on_data)]
fig, ax = plt.subplots(1, 2, figsize = (14, 7))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.text(s = "Liczba Trenowan Modelu od Jego Uczenia i Walidacji",
         size = 18,
         fontweight = 'bold',
         fontname = 'monospace',
         color = colors_dark[1],
         y = 1,
         x = 0.28,
         alpha = 0.8)

sns.despine()
ax[0].plot(epochs,
           train_acc,
           marker = 'o',
           markerfacecolor = colors_green[2],
           color = colors_green[3],
           label = "Dokladnosc uczenia")

ax[0].plot(epochs,
           val_acc,
           marker = 'o',
           markerfacecolor = colors_red[2],
           color = colors_red[3],
           label = "Dokladnosc walidacji")

ax[0].legend(frameon = False)
ax[0].set_xlabel("Liczba Trenowan Modelu")
ax[0].set_ylabel("Dokladnosc")

sns.despine()
ax[1].plot(epochs,
           train_loss,
           marker = 'o',
           markerfacecolor = colors_green[2],
           color = colors_green[3],
           label = "Straty uczenia")

ax[1].plot(epochs,
           val_loss,
           marker = 'o',
           markerfacecolor = colors_red[2],
           color = colors_red[3],
           label = "Straty walidacji")

ax[1].legend(frameon = False)
ax[1].set_xlabel("Liczba Trenowan Modelu")
ax[1].set_ylabel("Straty na Uczeniu i Walidacji")

fig.savefig("Liczba_Trenowan_Modelu_od_Jego_Uczenia_i_Walidacji.jpg")

pred_model = model.predict(X_test)
pred_model = np.argmax(pred_model, axis = 1)
y_test_new = np.argmax(y_test, axis = 1)

fig, ax = plt.subplots(1, 1, figsize = (14, 7))
sns.heatmap(confusion_matrix(y_test_new, pred_model),
            ax = ax,
            xticklabels = labels,
            yticklabels = labels,
            annot = True,
            cmap = colors_green[::-1],
            alpha = 0.7,
            linewidths = 2,
            linecolor = colors_dark[3])

fig.text(s = "Mapa Cieplna Dokladnosci Modelu",
         size = 18,
         fontweight = 'bold',
         fontname = 'monospace',
         color = colors_dark[1],
         x = 0.28,
         y = 0.92,
         alpha = 0.8)

plt.savefig("Mapa_Cieplna_Dokladnosci_Modelu.jpg")


print(classification_report(y_test_new, pred_model))
print("Work time: ", t_stop-t_start)