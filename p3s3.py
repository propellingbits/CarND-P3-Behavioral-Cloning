#using generator

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

lines = []
center_angle = 0
centerAngleImgCount = 0
brightness = 1.0
angleZeroImages = []
angleNonZeroImages = []
#plt.imshow(np.random.rand(10, 10), interpolation='none')

angles = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        if line[3] == 'steering':
            continue
        center_angle = float(line[3])
        
        if center_angle >= -0.1 and center_angle <= 0.1:
            angleZeroImages.append(line)
        else:
            angleNonZeroImages.append(line)        

#skipping the headers
#lines = lines[1:]

shuffle(angleZeroImages)
shuffle(angleNonZeroImages)

lines = angleZeroImages[0:1000]
lines.extend(angleNonZeroImages[0:2200])

print ('length of lines')
print (len(lines))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def normalize(image):
    #return (image/255 - 0.5)
    return image / 127.5 - 1

def blur(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def cropImage(img):
    return image[15:40, 0:80] #height, width, color channels

def resizeImage(img):
    #cv2.resize(img, (cols (width), rows (height)))
    img = cv2.resize(img, (80, 40))
    return img

def rgb2yuv(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def flipVertical(image):
    image = cv2.flip(image, 1)
    return image

def hsv(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    #print (image.shape)
    return image

def preprocessImage(img):
    #cv2.resize(img, (cols (width), rows (height)))
    ##img = cv2.resize(img, (200, 60))
    # img = cv2.resize(img, (80,40))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # img = img[15:40, 0:80]
    # #print (image.shape)
    # return img[:,:,1]
    # B,G,R channels of image index. We are locating grey channel 
    # #http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html -  image array 3 channel index   
    #img[:, :, 2] = img[:, :, 2] * brightness
    #img = normalize(blur(hsv(img)[:,:,1], kernel_size=5))
    #img = blur(hsv(img)[:,:,1], kernel_size=5)
    ##img = blur(img, kernel_size=5)
    croppedImage = cropImage(img)
    resizedImage = resizeImage(croppedImage)
    #rgb2yuved = rgb2yuv(resizedImage)
    #hsved = hsv(resizedImage)
    #blurred = blur(hsved, 5)
    normalizedImage = normalize(resizedImage)
    processedImg = normalizedImage
    return processedImg

x_train = [] #np.empty(shape=[0,3])
y_train = []

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
                    center_image = preprocessImage(cv2.imread(fileName))
                    center_angle = float(batch_sample[3])                    

                    images.append(center_image)
                    angles.append(center_angle)
                    
                    #if(center_angle > .30 or center_angle < -.30):
                    #    images.append(flipVertical(center_image))
                    #    angles.append(-center_angle)
                        #center_angle >= 0.01 or center_angle <= -.01
                
                    correction = 0.20 # this is a parameter to tune

                    if(center_angle > .10 or center_angle < -.10):
                        #left
                        fileName = "./data/IMG/" + batch_sample[1].split('/')[-1]
                        left_image = preprocessImage(cv2.imread(fileName))
                        left_angle = float(batch_sample[3]) + center_angle
                        
                        #if left_angle > 1.0:
                        #    left_angle = 1.0
                    #if (center_angle >= 0.01 or center_angle <= -.01):
                        images.append(left_image)
                        angles.append(left_angle)

                        #left flipped                    
                        #images.append(np.fliplr(left_image))
                        #angles.append(-left_angle)

                        #images.append(np.fliplr(left_image))
                        #angles.append(-left_angle)
                                        
                        fileName = "./data/IMG/" + batch_sample[2].split('/')[-1]
                        right_image = preprocessImage(cv2.imread(fileName))
                        right_angle = float(batch_sample[3]) - correction

                        #if right_angle < -1.0:
                        #    right_angle = -1.0
                        
                        images.append(right_image)
                        angles.append(right_angle)

                        #right flipped                    
                        #images.append(np.fliplr(right_image))
                        #angles.append(-right_angle)

                            
                            #angles = np.array([[angles]])

                        #images.append(np.fliplr(right_image))
                        #angles.append(-right_angle)
                except Exception as e : print (e)

            # trim image to only see section with road
            #print ("len:")
            #print (len(images)
            #print ('shape')
            #print (images.shape)
            x_train = np.array(images)
            y_train = np.array(angles)
            #images = np.array(np.reshape(images, (1, len(images), len(images[0]), 1)))
            #x_train = images.reshape(25, 80, 1, 1)
            #x_train = np.reshape( images, (None, 66, 200 , 3))
            #y_train = np.array(angles)
            #magic of yield. It takes a pause and release the data collected
            yield sklearn.utils.shuffle(x_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

#ch, row, col = 3,66,200

#todo - try keras image data generator



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Convolution1D, AveragePooling2D
from keras.layers import ELU
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Cropping2D, Cropping1D
from keras.optimizers import SGD, Adam, RMSprop

model = Sequential()
    
model.add(Convolution1D(64, 3, input_shape=(25, 80)))
model.add(Activation('relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
 validation_data=validation_generator, nb_val_samples=len(validation_samples),
  nb_epoch=6)

model.save('model-skinny.h5')
model.summary()
print('Model Created')