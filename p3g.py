#using generator

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        #range (start, stop, step)
        for offset in range(0, num_samples, batch_size):
            # goes from offset (start) to end (offset+batch_size)
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # -1 means last item
                try:
                    fileName = "./data/IMG/" + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(fileName)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                    
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                    
                    correction = 0.2 # this is a parameter to tune

                    #left
                    fileName = "./data/IMG/" + batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(fileName)
                    left_angle = float(batch_sample[3]) + correction
                    images.append(left_image)
                    angles.append(left_angle)

                    images.append(np.fliplr(left_image))
                    angles.append(-left_angle)

                    #right
                    fileName = "./data/IMG/" + batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(fileName)
                    right_angle = float(batch_sample[3]) - correction
                    images.append(right_image)
                    angles.append(right_angle)

                    images.append(np.fliplr(right_image))
                    angles.append(-right_angle)
                except Exception as e : print (batch_sample)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #magic of yield. It takes a pause and release the data collected
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


'''model = Sequential()

# Normalizing lambda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# First convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

# Second convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())

# Fully-connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))'''


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=10)

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')

print('Model Created')