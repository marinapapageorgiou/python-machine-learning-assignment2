from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy

from utils import load_cifar10

if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10(num_classes=3)


    # Preprocessing
    # scale pixels
    # convert from integers to floats
    train_norm = x_train.astype('float32')
    test_norm = x_test.astype('float32')
    # normalize each pixel of each channel so that the range is [0,1]
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # one-hot encoding of the labels
    trainY = to_categorical(y_train, 3)
    testY = to_categorical(y_test, 3)


    # Load the trained models (one or more depending on task and bonus)
    # for example
    model_task2= load_model('./nn_task2.h5')


    # Predict on the given samples
    # for example
    y_pred_task2 = model_task2.predict(test_norm)


    # Evaluate the missclassification error on the test set
    # for example
    assert testY.shape == y_pred_task2.shape
    acc = CategoricalAccuracy()
    acc.update_state(testY, y_pred_task2)
    acc2 = acc.result()
    print("Accuracy model task 1:", float(acc2))
