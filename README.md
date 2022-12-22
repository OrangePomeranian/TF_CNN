# TF_CNN

More informations about this project in Projekt Zaliczeniowy PDF.pdf

We processed 3200 MRI images of brain cancer with different classifications.

Our aim was to create a classification model for patients with suspected brain tumour development based on MRI images.

The project conisted of below points:
1. Import of libraries and modules
2. Data preparation
3. Dividing our data into testing and learning sets
4. Downloading the EfficienNetV2B0 model
5. Setting up the downloaded model accordingly
6. Compilation of the model
7. Appropriate setting of the reduction function 'leaning rate'
8. Training the model
9. Making a prediction


 
Examples of the functions we use from the Keras package from the TensorFlow package:

- tf.keras.applications.EfficientNetV2B0 
- tf.keras.callbacks.TensorBoard
- tf.keras.callbacks.ModelCheckpoint
- tf.keras.callbacks.ReduceLROnPlateau
- tf.keras.Model.fit()

Finally we presented the results with the use of seaborn library - Simple charts with accuracy of learning and validation, training loss and validation and heatmaps with classification of each tumour. 
![Screenshot 2022-12-22 at 20 12 22](https://user-images.githubusercontent.com/67764136/209210494-6b27e4c5-3fe2-49b4-a303-2096ecc2af29.png)
![Screenshot 2022-12-22 at 20 12 38](https://user-images.githubusercontent.com/67764136/209210482-c7341629-c846-417a-9e0c-993b0075cb98.png)



