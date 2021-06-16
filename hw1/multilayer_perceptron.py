import pandas as pd
import numpy as np
import os
import sys
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from util import compute_accuracy,load_data

torch.manual_seed(42)
""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_num, hidden_size, activation):
        super().__init__()
        modules = []
        activation_choice = None
        if activation == "relu":
            activation_choice = nn.ReLU()
        elif activation == "sigmoid":
            activation_choice = nn.Sigmoid()
        elif activation == "tanh":
            activation_choice = nn.Tanh()
        elif activation == "leaky relu":
            activation_choice = nn.RReLU()
            
        
        modules.append(nn.Linear(input_dim, hidden_size))
        modules.append(activation_choice)
        for i in range(hidden_num-1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(activation_choice)
        modules.append(nn.Linear(hidden_size, output_dim))
        modules.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*modules)
        

    def forward(self, x):
        return self.layers(x)

    
class MultilayerPerceptronModel(nn.Module):
    def __init__(self, hidden_num =1, hidden_size = 10, activation = "relu", learning_rate = 1e-4,  batch_size = 100, optimizer = "Adam", epoch = 10):
        super().__init__()
        self.hidden_num = hidden_num
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer
        self.model = None
        self.activation =activation
    def train(self, training_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
              
              a training example represented as an (input, label) pair.
        """
        training_feature = training_data[0] 
        train_label = training_data[1]
        print(training_feature.shape, train_label.shape)
        self.feature_dim = training_feature.shape[1]
        self.output_dim = train_label.shape[1]
        #define the MLP structure        
        self.model = MLP(self.feature_dim, self.output_dim, self.hidden_num, self.hidden_size,self.activation)
        #set up the training dataloader
        tensor_train_feature = torch.tensor(training_feature)
        tensor_train_label = torch.tensor(np.argmax(train_label,axis=1))
        train_dataset = TensorDataset(tensor_train_feature,tensor_train_label) 
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0 )

        #define the loss, using the cross-entropy loss
        loss_function = nn.CrossEntropyLoss()
        #define optimizer
        if self.optimizer  == "Adam":
            #lr=self.learning_rate
            optimizer = torch.optim.Adam(self.model.parameters())
        elif self.optimizer == "AdaGrad":
            #, lr=self.learning_rate
            optimizer = torch.optim.Adagrad(self.model.parameters())
        elif self.optimizer == "Adadelta" :   
            optimizer = torch.optim.Adadelta(self.model.parameters())
        #set up epoch training   
        for epoch in range(self.epoch):
            current_loss = 0.0    
            #Iterate over the DataLoader for training data by batch 
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs.float()) #
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
#             print("Loss for epoch %s is %s"%(epoch + 1, current_loss/self.batch_size))
        print('Training process has finished.')

    def predict(self, model_input):
        #convert to tensor
        tensor_feature = torch.tensor(model_input)
        output = self.model(tensor_feature.float())
        #output the label indexs as an tensor array
        return torch.argmax(output, axis = 1)
"""
How to use:
model = MultilayerPerceptronModel(hidden_num =i, hidden_size = j, learning_rate = 1e-4,batch_size = l, optimizer = "Adam", epoch = p)
model.train(train_data_prepared) #input a tupe of training feature and encoded label
_, predicted = model.predict(dev_feature) #input the development / any feature matrix
"""
#MLP evaluation on development set
def prepare_trainable_data(dataset, N, MinFreq, model_choice):
    train_data, dev_data, test_data,data_type = load_data(dataset, N, MinFreq, model_choice)
    train_feature = train_data[:,:-1]
    dev_feature = dev_data[:,:-1]
    train_feature = train_feature.astype("float64")
    dev_feature = dev_feature.astype("float64")
    train_label = train_data[:,-1]
    dev_label = dev_data[:, -1]
    all_label = np.hstack((train_label,dev_label))
    unique_label = np.unique(all_label)

    train_encoded_label = np.zeros((len(train_data),len(unique_label) ))
    dev_encoded_label = np.zeros((len(dev_data),len(unique_label) ))
    for i in range(len(train_label)):
        idx = np.where(unique_label == train_label[i])[0]
        train_encoded_label[i][idx] = 1
    for i in range(len(dev_label)):
        idx = np.where(unique_label == dev_label[i])[0]
        dev_encoded_label[i][idx] = 1

    train_data_prepared = (train_feature, train_encoded_label)
    dev_data_prepared = (dev_feature , dev_encoded_label)
    return train_data_prepared, dev_data_prepared

def MLP_dev_evaluation(train_data_prepared, dev_data_prepared, model): #model is an MultilayerPerceptronModel object
    model.train(train_data_prepared)
    dev_feature = dev_data_prepared[0]
    dev_label = np.argmax(dev_data_prepared[1], axis = 1)
    predicted = model.predict(dev_feature)
    predicted_label = np.array(predicted)

    dev_accuracy = compute_accuracy(predicted_label,dev_label)
    return dev_accuracy
#Hyper-parameter tuning for MLP on
#Number of hidden layers
#hidden layer size (assume each hidden layer has the same size)
#batch size
#Number of epocs
hidden_num = [i for i in range(1,7)]
hidden_size = [5*i for i in range(1,4)]
batch_size = [100 *i for i in range(1,9)]
epochs = [50 * i for i in range(1,8)]
def tune_hidden_batch_epochs(hidden_num , hidden_size , batch_size , epochs,train_data_prepared, dev_data_prepared):
    for i in hidden_num:
        for j in hidden_size:
            for l in batch_size:
                for p in epochs:
                    print("hidden layer: ",i)
                    print("hidden neurons: ",j)
                    print("batch size: ",l)
                    print("epoch: ",p)
                    model = MultilayerPerceptronModel(hidden_num =i, hidden_size = j, learning_rate = 1e-4,batch_size = l, optimizer = "Adam", epoch = p)
                    accuracy = MLP_dev_evaluation(train_data_prepared, dev_data_prepared, model)
                    print("development set accuracy: ",accuracy)
"""
For example, tuning parameters for propernames, (1,1)-gram MinFreq = 15
train_data_prepared, dev_data_prepared = prepare_trainable_data("propernames", (1,1), 15, "Combined")
tune_hidden_batch_epochs(hidden_num , hidden_size , batch_size , epochs,train_data_prepared, dev_data_prepared)
"""
# train_data_prepared, dev_data_prepared = prepare_trainable_data("propernames", (1,4),5, "Combined")
# model = MultilayerPerceptronModel(hidden_num =1, hidden_size = 10, learning_rate = 1e-4,batch_size = 500, optimizer = "Adam", epoch = 200)
# dev_accuracy = MLP_dev_evaluation(train_data_prepared, dev_data_prepared, model)
# print(dev_accuracy)

#Hyper-parameter tuning for MLP on
#Optimizer
#Activation function
optimizer_col = ["Adam","AdaGrad","Adadelta"]
activation_col = ["relu","sigmoid","tanh","leaky relu"]
# train_data_prepared, dev_data_prepared = prepare_trainable_data("propernames", (1,4), 5, "Combined")
# tune_hidden_batch_epochs(hidden_num , hidden_size , batch_size , epochs,train_data_prepared, dev_data_prepared)
# train_data_prepared, dev_data_prepared = prepare_trainable_data("newsgroups", (1,2), 40, "Combined")
def tune_optimizer_activation(optimizer_col, activation_col,train_data_prepared, dev_data_prepared):
    for i in optimizer_col:
        for j in activation_col:
            
            print("Optimizer: ",i)
            print("Activation: ",j)
                  
            model = MultilayerPerceptronModel(hidden_num =1, hidden_size = 10, learning_rate = 1e-4,batch_size = 300, optimizer = i,activation =j, epoch =200)
            accuracy = MLP_dev_evaluation(train_data_prepared, dev_data_prepared, model)
            print("development set accuracy: ",accuracy)
"""
For example, tuning optimizer and activation for propernames on 4-gram, Minimal frequency = 5, with 1 hidden layer, 10 hidden neurons, 300 batch size, and 200 training epochs
train_data_prepared, dev_data_prepared = prepare_trainable_data("propernames", (1,4), 5, "Combined")
tune_optimizer_activation(optimizer_col, activation_col,train_data_prepared, dev_data_prepared)
"""
