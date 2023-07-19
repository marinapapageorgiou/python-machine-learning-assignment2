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
trainY = to_categorical(trainY, 3)
testY = to_categorical(testY, 3)


# STEP_3
def define_model(lr, n):
    model = Sequential()
    # convolutional layer, with 8 filters of size 5×5, stride of 1×1, and ReLU activation
    model.add(Conv2D(8, (5, 5), strides = (1, 1), activation="relu"))
    # max pooling layer, with pooling size of 2×2
    model.add(MaxPooling2D((2, 2)))
    # convolutional layer, with 16 filters of size 3×3, stride of 2×2, and ReLU activation
    model.add(Conv2D(16, (3, 3), strides = (2, 2), activation="relu"))
    # average pooling layer, with pooling size of 2×2
    model.add(AveragePooling2D(pool_size = (2, 2)))
    # layer to convert the 2D feature maps to vectors (Flatten layer)
    model.add(Flatten())
    # dense layer with 8 neurons and tanh activation
    model.add(Dense(n, activation="tanh"))
    # dense output layer with softmax activation
    model.add(Dense(3, activation="softmax"))


    # STEP_4
    # plot diagnostic learning curves = compilation
    # compile model
    opt = RMSprop(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# define cnn model
model = define_model(0.003, 8)

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


# STEP_7 - BONUS PART
# define cnn model
learning_rate = [0.01,0.0001]
number_of_neurons = [16,64]
for i in learning_rate:
    for j in number_of_neurons:
        new_model = define_model(i,j)
        # fitting
        new_history = new_model.fit(
            train_norm,
            trainY,
            epochs=500,
            validation_split=0.2,
            batch_size=128,
            callbacks=[early_stopping],
            # make the function talkative
            verbose=0
        )
        # evaluate my train test
        new_eval, new_acc = new_model.evaluate(test_norm, testY, verbose=1)
        print("The accuracy of lr+"+str(i)+" and neurons="+str(j)+" is: " + str(new_acc))


# save my data to the file nn_task1.h5
save_keras_model(model, '../deliverable/nn_task1.h5')


"""
94/94 [==============================] - 0s 3ms/step - loss: 0.3825 - accuracy: 0.8543
The accuracy of current model is: 0.8543333411216736

94/94 [==============================] - 0s 3ms/step - loss: 0.4115 - accuracy: 0.8387
The accuracy of lr+0.01 and neurons=16 is: 0.8386666774749756
94/94 [==============================] - 0s 3ms/step - loss: 0.4440 - accuracy: 0.8190
The accuracy of lr+0.01 and neurons=64 is: 0.8190000057220459
94/94 [==============================] - 0s 3ms/step - loss: 0.4673 - accuracy: 0.8123
The accuracy of lr+0.0001 and neurons=16 is: 0.812333345413208
94/94 [==============================] - 0s 4ms/step - loss: 0.4359 - accuracy: 0.8240
The accuracy of lr+0.0001 and neurons=64 is: 0.8240000009536743
"""
