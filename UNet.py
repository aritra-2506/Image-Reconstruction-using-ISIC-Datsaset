import cv2
import glob
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import matplotlib.pyplot as plt

#Load image files from directory
filenames = glob.glob("C:/Users/Aritra Mazumdar/Downloads/ISIC/img/*.jpg")
filenames.sort()
images_list = [cv2.imread(img) for img in filenames]

#Data Preprocessing - Resize & Rescale images
X=[]
for img in images_list:
    X.append(cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))
images=np.asarray(X)
images = images.astype('float32') / 255

#Train-Test split
images_train=images[0:4000]
images_val=images[4000:5000]
images_test=images[5000:6007]

#Buliding Network
input_img = Input(shape=(256, 256, 3))

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(0.25)(pool1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(0.5)(pool2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(0.5)(pool3)

conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.5)(pool4)

convm = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
convm = Conv2D(1024, (3, 3), activation='relu', padding='same')(convm)

deconv4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = concatenate([deconv4, conv4])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(uconv4)

deconv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = concatenate([deconv3, conv3])
uconv3 = Dropout(0.5)(uconv3)
uconv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv3)

deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = concatenate([deconv2, conv2])
uconv2 = Dropout(0.5)(uconv2)
uconv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv2)

deconv1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = concatenate([deconv1, conv1])
uconv1 = Dropout(0.5)(uconv1)
uconv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv1)

ouput_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(uconv1)

autoencoder = Model(input_img, ouput_img)

#Compiling
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting
autoencoder.fit(images_train, images_train,
epochs=40,
batch_size=50,
shuffle=True,
validation_data=(images_val, images_val),
)

plot_model(model, to_file='model.png')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#Plot Accuracy
f=plt.figure(1)
plt.plot(autoencoder.history['acc'])
plt.plot(autoencoder.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
f.show()

#Plot Loss
g=plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')

plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
g.show()

y_pred = model.predict(test_images)

plt1.figure()
plt.subplot(2, 1, 1)
plt.title('Original Image')
plt.imshow(test_images[100])
plt.subplot(2, 1, 2)
plt.title('Reconstructed Image')
plt.imshow(y_pred[100])
