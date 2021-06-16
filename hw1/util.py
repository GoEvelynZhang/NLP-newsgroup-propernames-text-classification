from propername import propername_data_loader
from newsgroup import newsgroup_data_loader
import pandas as pd
import numpy as np

def save_results(predictions, results_path):
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    dic = {"Predicted": predictions}
    df = pd.DataFrame(dic)
    df.to_csv(results_path, index = False)
    # TODO: Implement saving of the results.
#     pass

def compute_accuracy(labels, predictions):
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as " \
                         "length of predictions (" + str(len(predictions)) + ".")
    acc = sum(1 for x,y in zip(labels,predictions) if x == y) / len(predictions)

    return acc

def evaluate(model, data, results_path):
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    """
    feature = data[0]
    feature_bias = np.insert(feature, 0, 1, axis=1)
    print(feature_bias.shape, model.weight.shape)
    predictions = model.predict(feature_bias)
    true_label = np.argmax(data[1],  axis=1)
    

    save_results(predictions, results_path)

    return compute_accuracy(true_label, predictions)

def load_data(args, N, min_freq, model_choice):
    """ Loads the data.

    Inputs:
        args (list of str): The command line arguments passed into the script.

    Returns:
        Training, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    data_type = ""
    if 'propernames' in args:
        data_loader = propername_data_loader
        data_type = "propernames"
    elif 'newsgroups' in args:
        data_loader = newsgroup_data_loader
        data_type = "newsgroups"
    assert data_loader, "Choose between newsgroup or propername data. " \
                        + "Args was: " + str(args)

    # Load the data. 
    train_data, dev_data, test_data = data_loader("data/" + data_type + "/train/train_data.csv",
                                                  "data/" + data_type + "/train/train_labels.csv",
                                                  "data/" + data_type + "/dev/dev_data.csv",
                                                  "data/" + data_type + "/dev/dev_labels.csv",
                                                  "data/" + data_type + "/test/test_data.csv",N, min_freq, model_choice)

    return train_data, dev_data, test_data, data_type
