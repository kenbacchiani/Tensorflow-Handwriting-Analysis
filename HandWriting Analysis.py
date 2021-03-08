#Ken Bacchiani
#Handwriting Analysis.py
#Written for introduction to Artificial Intelligence course at University
#of Wisconsin

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_dataset(training=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if(training == True):
        return(train_images, train_labels)
    else:
        return(test_images, test_labels)


def print_stats(train_images, train_labels):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    print(len(train_images))
    print(len(train_images[0]),"x",len(train_images[0][0]), sep = '')
    values = [0] * 10
    for i in range(len(train_labels)):
        values[train_labels[i]] += 1
    for i in range(len(class_names)):
        print(i,". ", class_names[i], " - ", values[i], sep = '')


def build_model():
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    opt = keras.optimizers.SGD(learning_rate=0.001)
    model.compile(
    optimizer= opt,
    metrics=['accuracy'],
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)
    return(model)


def train_model(model, train_images, train_labels, T) :
    model.fit(x = train_images, y = train_labels, epochs = T, verbose = 1)

    
def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 0)
    if(show_loss == True):
        print('Loss:', "{:.4f}".format(test_loss))
    percent = test_accuracy * 100
    print('Accuracy: ', "{:.2f}".format(percent), "%", sep = '')

    
def predict_label(model, test_images, index):
    guess = model.predict(x = test_images)
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    sorted = np.argsort(guess[index])
    big1 = sorted[-1]
    big2 = sorted[-2]
    big3 = sorted[-3]
    print(class_names[big1], ": ", '%.2f'%(100 * guess[index][big1]), "%",  sep = '')
    print(class_names[big2], ": ", '%.2f'%(100 * guess[index][big2]), "%", sep = '')
    print(class_names[big3], ": ", '%.2f'%(100 * guess[index][big3]), "%", sep = '')
                                                           
                                                             
if __name__=="__main__":    
    (train_images, train_labels) = get_dataset()
    (test_images, test_labels) = get_dataset(False)
    model = build_model()
    train_model(model, train_images, train_labels, 10)
    model.add(tf.keras.layers.Softmax())
    opt = keras.optimizers.SGD(learning_rate=0.001)
    model.compile(
    optimizer= opt,
    metrics=['accuracy'],
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)
    predict_label(model, test_images, 2)
