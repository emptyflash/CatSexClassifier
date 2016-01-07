from transform_data import (get_input_images_and_ouput_labels,
                            get_number_of_image_files_in_path)
from config import *

import sys
import random
import time
from itertools import izip, cycle

import numpy as np

import theano
import theano.tensor as T

import cPickle as pickle
import os

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.nonlinearities import sigmoid


def generate_minibatches(generator, batch_size):
    batch = [(X, Y) for _, (X, Y) in izip(xrange(batch_size), generator)]
    return batch


def get_input_and_output_from_batch(batch):
    # The slice here is to make sure we only have 3 channels
    X = np.array([X[:3, :, :] for X, _ in batch], dtype=theano.config.floatX)

    # Randomly flip images
    batch_size = X.shape[0]
    indices_to_flip = np.random.choice(
        batch_size, batch_size / 2, replace=False)
    X[indices_to_flip] = X[indices_to_flip, :, :, ::-1]

    Y = np.vstack([Y for _, Y in batch]).astype(np.int8)
    return X, Y


def get_percentage_of_generator(generator, number_of_items, batch_size, percent):
    return ((X, Y) for _, (X, Y) in izip(
        xrange(int(number_of_items * percent / batch_size) * batch_size),
        generator))


def build_neural_network(input_var, input_shape):
    net = {}

    net['input'] = InputLayer(input_shape, input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=7,
                             stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
    net['pool1'] = PoolLayer(net['norm1'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['conv4'] = ConvLayer(net['conv3'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['conv5'] = ConvLayer(net['conv4'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], sigmoid)

    return net


def read_model_data(model, filename):
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    data = lasagne.layers.get_all_param_values(model)
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def main(num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    input_var = T.tensor4('inputs')
    target_var = T.bmatrix('targets')

    network = build_neural_network(input_var, (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE))['prob']

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss,
                                       params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                       target_var)
    test_loss = test_loss.mean()

    test_accuracy = T.mean(T.eq(T.gt(test_prediction, 0.5), T.eq(target_var, 1.0)),
                           dtype=theano.config.floatX)

    train_function = theano.function([input_var, target_var],
                                     loss,
                                     updates=updates)

    validation_function = theano.function([input_var, target_var],
                                          [test_loss, test_accuracy])

    number_of_image_files = get_number_of_image_files_in_path()

    print "Number of files found: ", number_of_image_files

    print("Starting training...")

    # Move this to the epoch loop and remove the cycle for lower memory computers
    data_generator = cycle(get_input_images_and_ouput_labels())
    best_accuracy = 0
    for epoch in range(num_epochs):
        train_generator = get_percentage_of_generator(data_generator,
                                                      number_of_image_files,
                                                      batch_size,
                                                      0.6)
        validation_generator = get_percentage_of_generator(data_generator,
                                                           number_of_image_files,
                                                           batch_size,
                                                           0.2)
        test_generator = get_percentage_of_generator(data_generator,
                                                     number_of_image_files,
                                                     batch_size,
                                                     0.2)

        # In each epoch, we do a full pass over the training data:
        train_error = 0
        train_batches = 0
        start_time = time.time()
        while True:
            try:
                batch = generate_minibatches(train_generator, batch_size)
            except StopIteration:
                break
            if len(batch) != batch_size:
                break
            X, Y = get_input_and_output_from_batch(batch)
            train_error += train_function(X, Y)
            train_batches += 1

        print "Finished training for epoch {} with a total of {} batches".format(
            epoch + 1, train_batches)

        # And a full pass over the validation data:
        val_error = 0
        val_accuracy = 0
        val_batches = 0
        while True:
            try:
                batch = generate_minibatches(validation_generator, batch_size)
            except StopIteration:
                break
            if len(batch) != batch_size:
                break
            X, Y = get_input_and_output_from_batch(batch)
            err, acc = validation_function(X, Y)
            val_error += err
            val_accuracy += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_error / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_error / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_accuracy / val_batches * 100))

        print("  train / valid:\t\t{:.6f}".format(
            (train_error / train_batches) / (val_error / val_batches)))

        if val_accuracy > best_accuracy:
            print "Accuracy better than previous best, saving model"
            write_model_data(network, "models/model%s.pkl" % int(val_accuracy / val_batches * 100))
            best_accuracy = val_accuracy

    # After training, we compute and print the test error:
    test_error = 0
    test_accuracy = 0
    test_batches = 0
    while True:
        try:
            batch = generate_minibatches(test_generator, batch_size)
        except StopIteration:
            break
        if len(batch) != batch_size:
            break
        X, Y = get_input_and_output_from_batch(batch)
        err, acc = validation_function(X, Y)
        test_error += err
        test_accuracy += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_error / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_accuracy / test_batches * 100))

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    main(**kwargs)
