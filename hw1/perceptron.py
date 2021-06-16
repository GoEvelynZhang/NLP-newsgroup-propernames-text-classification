""" Perceptron model for Assignment 1 """
import os
import sys
import numpy as np
from util import evaluate, load_data
import time

class PerceptronModel():
    """ Perceptron for classification.

    Attributes:

    """
    def __init__(self, in_dim, classnames, n_epoch, bias):
        # Initialize the parameters of the model.
        # in_dim: Number of input features (number of n-grams etc.)
        # out_dim: Number of output classes (5 for properNames, 20 for newsGroups)
        self.n_epoch = n_epoch
        self.in_dim = in_dim
        self.bias = bias
        if self.bias == 'bias':
            self.in_dim += 1
        self.classnames = classnames
        self.weight = {}
        # self.class_id = list(range(self.out_dim))
        for c in self.classnames: #create one weight vector for each class
            self.weight[c] = [0] * self.in_dim

    def train_long(self, training_data):
        """ Trains the perceptron model for specified epochs without evaluation in the middle (for speed training).

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        self.n_samples = len(training_data)
        for epoch in range(self.n_epoch):
            print("Epoch {}".format(epoch))
            # Loop through training data
            for i in range(self.n_samples):
                X,y_true = training_data[i] # TODO: Need to modify depending on input format
                # print('Sample {}'.format(i))
                # print(X.shape)
                y_pred = self.predict(X)
                if y_pred != y_true:
                    self.weight[y_pred] -= X
                    self.weight[y_true] += X

    def train(self, training_data):
        """ Trains one epoch of the perceptron model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        self.n_samples = len(training_data.keys())
        for i in list(training_data.keys()):
            X,y_true = training_data[i]
            if self.bias == 'bias':
                X.append(self.in_dim - 1)
            # print('Sample {}'.format(i))
            y_pred = self.predict(X)
            if y_pred != y_true:
                for idx in X:
                    self.weight[y_pred][idx] -= 0.01
                    self.weight[y_true][idx] += 0.01

    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (list): List of ids corresponding to the unique n-grams present in the document.

        Returns:
            The predicted class.

        """
        preds = {}
        for c in self.classnames:
            preds[c] = np.sum([self.weight[c][idx] for idx in model_input])#np.dot(self.weight[c], model_input)
        preds_out = list(preds.values())
        id_max = preds_out.index(max(preds_out))
        return list(preds.keys())[id_max] # return class id of the top scoring class

if __name__ == "__main__":
    # Param
    task_name = sys.argv[1]
    n_epoch = int(sys.argv[2])
    feat_type = sys.argv[3]
    N1 = int(sys.argv[4])
    N2 = int(sys.argv[5])
    min_freq = int(sys.argv[6])
    eval_every = int(sys.argv[7])
    presaved = sys.argv[8]
    bias = sys.argv[9]
    patience = 50//eval_every
    # Load data
    print('Loading Data ...')
    start = time.time()
    # Dataset loaded as a dictionary of tuples. Each sample is saved in the form (word_id, label).
    train_data, dev_data, test_data, data_type, in_dim, classnames = load_data(task_name,feat_type, N1, N2, min_freq, presaved)
    end = time.time()
    print('(Time elapsed: {})'.format(end - start))
    print('# Number of features: {}'.format(in_dim))
    # Train the model using the training data.
    print('Training {} with {} feature with N={},{} ...'.format(task_name,feat_type, N1, N2))
    model = PerceptronModel(in_dim, classnames, n_epoch,bias)
    best_acc = 0
    no_improve_counter = 0

    train_acc_list = []
    val_acc_list = []
    train_acc_list.append('{}_{}_{}'.format(feat_type, N1, N2))
    val_acc_list.append('{}_{}_{}'.format(feat_type, N1, N2))

    for epoch in range(model.n_epoch):
        print("Epoch {}".format(epoch))
        model.train(train_data)

        if epoch%eval_every == 0:
            # Predict on the train set.
            train_acc = evaluate(data_type,model,
                                train_data,
                                os.path.join("results", "perceptron_" + data_type.rstrip('s') + "_train_predictions.csv"),
                                save_result = False)

            # Predict on the development set.
            dev_acc = evaluate(data_type,model,
                                dev_data,
                                os.path.join("results", "perceptron_" + data_type.rstrip('s') + "_dev_predictions.csv"),
                               save_result = False)
            train_acc_list.append(str(train_acc))
            val_acc_list.append(str(dev_acc))

            print('Accuracy on train set: {}, Accuracy on dev set: {}'.format(train_acc, dev_acc))

            no_improve_counter += 1

            if dev_acc > best_acc:
                best_acc = dev_acc
                no_improve_counter = 0 # reset counter

                # Predict on the test set using best validation model.
                evaluate(data_type, model,
                         test_data,
                         os.path.join("results", "perceptron_" + data_type.rstrip('s') + "_test_predictions.csv"))

            if no_improve_counter > patience:
                print('No improvement after {} epochs. Early stopping...'.format(no_improve_counter))
                break

    # log_path_train = 'log/log_{}_train_perceptron.csv'.format(task_name)
    # log_path_dev = 'log/log_{}_dev_perceptron.csv'.format(task_name)
    #
    # with open(log_path_train, 'a') as fd_train:
    #     fd_train.write(",".join(train_acc_list)+'\n')
    # fd_train.close()
    #
    # with open(log_path_dev, 'a') as fd_dev:
    #     fd_dev.write(",".join(val_acc_list)+'\n')
    # fd_dev.close()
