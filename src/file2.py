# STEP_1
# test harness for evaluating models on the cifar10 dataset
from cmath import tanh
import sys
# from matplotlib import pyplot
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.optimizers.optimizer_v2.rmsprop import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from utils import load_cifar10, save_keras_model

# load dataset
(trainX, trainY), (testX, testY) = load_cifar10(3)


# STEP_2
# scale pixels
# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
# normalize each pixel of each channel so that the range is [0,1]
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
# one-hot encoding of the labels
trainY = to_categorical(trainY)
testY = to_categorical(testY)


# STEP_3
model = Sequential()
# flatten the images into 1D vectors (Flatten layer)
model.add(Flatten())
# use only dense layers - hidden layer
# use ReLU activation for all layers other than the output one - put relu as an activation parameter
model.add(Dense(64, activation="relu"))
# use no more than 3 layers, considering also the output one
model.add(Dense(16, activation="relu"))
# use Softmax activation for the - output layer
model.add(Dense(3, activation="softmax"))


# STEP_4
# plot diagnostic learning curves = compilation
# compile model
opt = RMSprop(learning_rate=0.003)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(
    train_norm,
    trainY,
    epochs=500,
    validation_split=0.2,
    batch_size=128,
    callbacks=[early_stopping],
    # make the function talkative
    verbose=1
)


# STEP_5
# plot with epochs on the x-axis the train accuracy
# plot with epochs on the x-axis the validation accuracy
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
plt.plot(loss_train, 'g', label='Training accuracy')
plt.plot(loss_val, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# STEP_6
# run the test harness for evaluating a model
# evaluate my train test
eval, acc = model.evaluate(test_norm,testY,verbose=1)
print("The accuracy of current model is: "+str(acc))


# save my data to the file nn_task1.h5
save_keras_model(model, '../deliverable/nn_task2.h5')

