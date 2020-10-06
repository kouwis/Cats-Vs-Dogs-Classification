#imports
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
import numpy as np
import cv2

#Initialize the training and vali Gen
train_dir = "train"
vali_dir = "validation"

train_datagen = ImageDataGenerator(rescale=1./255)
vali_datagen = ImageDataGenerator(rescale=1./255)

train_Gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True
)
vali_Gen = vali_datagen.flow_from_directory(
    vali_dir,
    target_size=(128, 128),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True
)

#Build model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(128,128,3), activation="relu"))
model.add((MaxPooling2D(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add((MaxPooling2D(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add((MaxPooling2D(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(2, activation="sigmoid"))

model.summary()

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

model_fit = model.fit_generator(
    train_Gen,
    steps_per_epoch=2000//64,
    epochs=50,
    validation_data=vali_Gen,
    validation_steps=5000//64
)

#Save weights
model.save("CatsVsDogs.h5")

#Load weights
model = load_model("CatsVsDogs.h5")

#Classifier
def Image_Classification(ImgPath):

    aa = {0: "cat", 1: "dog"}
    img = cv2.imread(ImgPath)
    img2 = np.copy(img)
    img2 = cv2.resize(img, (480,360))
    img = img/255
    img = cv2.resize(img,(128,128))
    img = np.reshape(img,[1,128,128,3])
    classes = model.predict(img)
    classes = np.argmax(classes)
    cv2.putText(img2, aa[classes], (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img2

img = Image_Classification("ImgPath")
cv2.imshow("Classifier",img)
cv2.waitKey(0)
cv2.destroyAllWindows()