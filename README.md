# TF_CNN

More informations about this project in Projekt Zaliczeniowy PDF.pdf

We processed 3200 MRI images of brain cancer with different classifications.

Our aim was to create a classification model for patients with suspected brain tumour development based on MRI images.

The project consisted of below points:
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
![](https://github.com/Michello077/tumor-classification-using-CNN/blob/4344f59c1d397dc91127bbef2bf6876a89cf84c7/results/CNN001.png)
![](https://github.com/Michello077/tumor-classification-using-CNN/blob/4344f59c1d397dc91127bbef2bf6876a89cf84c7/results/CNN002.png)



