import domino_copy_theanorc
from transform_data import (get_input_images_and_ouput_labels,
                            get_number_of_image_files_in_path)

import sys
import random
import time
from itertools import izip

from numpy import array, vstack, int8, float32

import theano
import theano.tensor as T

import cPickle as pickle
import os

import lasagne


def generate_minibatches(generator, batch_size=50):
    batch = [(X, Y) for _, (X, Y) in izip(xrange(batch_size), generator)]
    # batch.extend([(X[:, :, ::-1], Y) for (X, Y) in batch])
    random.shuffle(batch)
    return batch


def get_input_and_output_from_batch(batch):
    # The slice here is to make sure we only have 3 channels
    X = array([X[:3, :, :] for X, _ in batch], dtype=theano.config.floatX)
    Y = vstack([Y for _, Y in batch]).astype(int8)
    return X, Y


def build_neural_network(input_var, input_shape):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network,
                                         num_filters=32,
                                         filter_size=(5, 5),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(network,
                                         num_filters=32,
                                         filter_size=(5, 5),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=1,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    return network


def read_model_data(model, filename):
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    data = lasagne.layers.get_all_param_values(model)
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def main(num_epochs=500):
    input_var = T.tensor4('inputs')
    target_var = T.bmatrix('targets')

    network = build_neural_network(input_var, (50, 3, 200, 200))

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss,
                                                params,
                                                learning_rate=0.01,
                                                momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                       target_var)
    test_loss = test_loss.mean()

    test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                           dtype=theano.config.floatX)

    train_function = theano.function([input_var, target_var],
                                     loss,
                                     updates=updates)

    validation_function = theano.function([input_var, target_var],
                                          [test_loss, test_accuracy])

    number_of_image_files = get_number_of_image_files_in_path()
    data_generator = get_input_images_and_ouput_labels()

    train_generator = ((X, Y) for _, (X, Y) in
                       izip(xrange(int(number_of_image_files * 0.6)),
                            data_generator))
    validation_generator = ((X, Y) for _, (X, Y) in
                            izip(xrange(int(number_of_image_files * 0.2)),
                                 data_generator))
    test_generator = ((X, Y) for _, (X, Y) in
                      izip(xrange(int(number_of_image_files * 0.2)),
                           data_generator))

    print("Starting training...")

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_error = 0
        train_batches = 0
        start_time = time.time()
        while True:
            try:
                batch = generate_minibatches(train_generator)
            except StopIteration:
                break
            X, Y = get_input_and_output_from_batch(batch)
            train_error += train_function(X, Y)
            train_batches += 1

        # And a full pass over the validation data:
        val_error = 0
        val_accuracy = 0
        val_batches = 0
        while True:
            try:
                batch = generate_minibatches(validation_generator)
            except StopIteration:
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

    # After training, we compute and print the test error:
    test_error = 0
    test_accuracy = 0
    test_batches = 0
    while True:
        try:
            batch = generate_minibatches(test_generator)
        except StopIteration:
            break
        X, Y = get_input_and_output_from_batch(batch)
        err, acc = validation_function(inputs, targets)
        test_error += err
        test_accuracy += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_error / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_accuracy / test_batches * 100))

    write_model_data(network, "models/model.pkl")

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    main(**kwargs)
