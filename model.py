import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import ELU

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# To be tuned
steer_correction = 0.1

# data set directory names
data_folder_names = [
                    'data',
                #    'my_driving_data_counter_clock',
                #    'my_driving_data_swerves',
		]

def random_steer_angle_shift(angle):
    """
    Add random steering shift
    """
    return (angle * (1.0 + np.random.uniform(-1, 1)/30.0)) 

def image_generator(samples, batch_size=32):
    """
    Data set generator
    """
    num_samples = len(samples)

    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            #print("Generating images")
            for batch_sample in batch_samples:
                for side in range(0,3):   # 0=center, 1=left, 2=right
                    #print("Fetching image: " + str(side))
                    source_path = batch_sample[side]
                    filename = source_path.split('/')[-1]
                    current_path = '.\\' + batch_sample[7] + '\IMG\\' + filename
                    #print("File being processed: " + current_path )
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    measurements.append(measurement)

	            # Flip the image and append except center camera images
                    if ( side != 0 ):
                        images.append(np.fliplr(image))
                        # Correct steering angle based on the side: center=no change,; left += steer_correction; right -= steer_correction
                        if (side == 1 ): measurement += steer_correction
                        if (side == 2 ): measurement -= steer_correction
                        measurements.append(random_steer_angle_shift(-measurement))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def buildModel():
    """
    Build model
    """

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def main():
    """
    Main function
    """

    print("In main now")
    #print(os.getcwd())

    lines = []
    # extrace lines out of csv files in different data folders.
    # 'data' contains ult udacity data
    # 'my_driving_data_counter_clock' contains counter-clock-wise driving data
    # 'my_driving_data_swerves' contains data focusing on recovery
    # Had to update a bit for running on Windows
    for folder in data_folder_names:
        with open('.\\' + folder + '\driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            # skip header row
            next(reader)
            for line in reader:
                line.append(folder)
                lines.append(line)

    train_set, validation_set = train_test_split(lines, test_size=0.2)

    # extract images and measurement array
    train_generator = image_generator(train_set, batch_size=32)
    validation_generator = image_generator(validation_set, batch_size=32)

    print("Begin training.....")

    # Build network here
    myModel = buildModel()

    myModel.summary()

    myModel.compile(loss='mse', optimizer='adam')
    history_object = myModel.fit_generator(train_generator, samples_per_epoch=len(train_set), validation_data=validation_generator, nb_val_samples=len(validation_set), nb_epoch=8, verbose=1)

    myModel.save('model.h5')

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
