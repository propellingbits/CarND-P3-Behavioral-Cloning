import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

lines = lines[1:]

for line in lines:
	source_path = line[0]
	tokens = source_path.split('/')
	filename = tokens[-1]
	local_path = "./data/IMG/" + filename
	img_center = cv2.imread(local_path)
	images.append(img_center)
	steering_center = float(line[3])
	measurements.append(steering_center)
	
	correction = 0.2 # this is a parameter to tune

	#left
	source_path = line[1]
	tokens = source_path.split('/')
	filename = tokens[-1]
	local_path = "./data/IMG/" + filename
	img_left = cv2.imread(local_path)	
	steering_left = steering_center + correction
	images.append(img_left)
	measurements.append(steering_left)

	#right
	source_path = line[2]
	tokens = source_path.split('/')
	filename = tokens[-1]
	local_path = "./data/IMG/" + filename
	img_right = cv2.imread(local_path)
	steering_right = steering_center - correction
	images.append(img_right)
	measurements.append(steering_right)	

#skipping the first element
#X_train = np.array(images[1:])
#y_train = np.array(measurements[1:])

#images = images[1:]
#measurements = measurements[1:]

'''# Arrays to store additional images and measurements
augmented_images = []
augmented_measurements = []
images = images[1:]
measurements = measurements[1:]

for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	
	# Flip images to reduce bias from anti-clockwise driving
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

augmented_images = np.array(augmented_images)
augmented_measurements = np.array(augmented_measurements)

X_train = np.concatenate(X_train, augmented_images)
y_train = np.concatenate(y_train, augmented_measurements)'''

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	#flipped_image = cv2.flip(image, 1)
	flipped_image = np.fliplr(image)
	#measurement_flipped = -measurement
	flipped_measurement = -measurement #float(measurement) * -2.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')

print('Model Created')