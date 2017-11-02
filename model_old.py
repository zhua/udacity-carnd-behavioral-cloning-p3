import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# To be tuned
steer_correction = 0.1

def main():
	print("In main now")
	#print(os.getcwd())
  
	lines = []
	# extrace lines out of csv file. Had to update a bit for running on Windows
	with open('.\data\driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		# skip header row
		next(reader)
		for line in reader:
			lines.append(line)
	
	# extract images and measurement array
	images = []
	measurements = []
	# Get center/right/left images, flip them and save in arrays
	print("Parsing images")
	for line in lines:
		for side in range(0,3):   # 0=center, 1=left, 2=right
			#print("Fetching image: " + str(side))
			source_path = line[side]
			filename = source_path.split('/')[-1]
			current_path = '.\data\IMG\\' + filename
			#print("Image " + str(side) + ":  " + current_path)
			image = cv2.imread(current_path)
			images.append(image)
			measurement = float(line[3])
			measurements.append(measurement)
			
			# Flip the image and append
			images.append(np.fliplr(image))
			# Correct steering angle based on the side: center=no change,; left += steer_correction; right -= steer_correction
			if (side == 1 ): measurement += steer_correction
			if (side == 2 ): measurement -= steer_correction
			measurements.append(-measurement)


	print("Begin training.....")
	X_train = np.array(images)
	y_train = np.array(measurements)

	# Build network here
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	#model.add(Flatten(input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.summary()

	model.compile(loss='mse', optimizer='adam')
	history_object = model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=10)

	model.save('model.h5')

        ### print the keys contained in the history object
	print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

	print("Model training complete!")



if __name__ == "__main__":
	main()	
