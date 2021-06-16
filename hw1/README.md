# a1--group-5
a1--group-5 created by GitHub Classroom
# CS5740: Assignment 1

This repo implements Perceptron and MLP for the classificatoin of the Proper Name dataset and the 20 Newsgroups dataset.


## Prerequisites

- Python3
- torch 1.7.1
- pandas 1.2.1
- tokenizers 0.10.0
- ipython 7.20.0

## Usage for Perceptron

Sample command to call:
```
python perceptron.py dataset n_epochs n_gram1 ngram2 min_freq evaluate_every use_presaved_input use_bias
python perceptron.py propername 30 Combined 1 4 5 1 no_presaved no_bias
python perceptron.py newsgroup 30 Combined 1 4 5 1 presaved no_bias
```
## Preprocessing 
### NewsGroup Task Preprocessing 
* convert text to lower case
* remove url in text
* remove non-ascii in text
* remove punctuation
* remove stopwords
### Proper Task Preprocessing 
* convert text to lower case
* remove url in text
* remove non-ascii in text


## Task Specific feature engineering
### NewsGroup Task Specific feature engineering
* Each group is distinguished by whether headers such as NNTP-Posting-Host: and Distribution: appear more or less often.
* Another significant feature involves whether the sender is affiliated with a university, as indicated either by their headers or their signature
* The word “article” is a significant feature, based on how often people quote previous posts like this: “In article [article ID], [name] <[e-mail address]> wrote:”

So that we generate the following features after preprocessing the text data
* host_count: count the # of "NNTP-Posting-Host:" appearance in 
* distribution_count : count the # of "Distribution:"
* from_university: binary variable for whether email address of sender contains "edu"
* reference_count: count the # of "In article" in a text


### Proper name Task Specific feature engineering
* Text containing numbers can hardly be names
* Text containing some punctuation (&\?-:"%) can hardly be names
* Text containing "Inc." can be strog indicator for class company
* Text containing only one token can hardly be a proper name


So that we generate the following features after preprocessing the text data
* contain_numerics: binary variable whether text contains numericals
* contain_special_punc: binary variable whether text contains  some special punctuation (&\?-:"%)
* contain_inc: binary variable whether text contains "Inc." in the end 
* Small_token_length: binary variable whether text contains only taken length 1

## Hyper-parameter tuning in MLP
The MLP class is set up flexibly to tune number of hidden layers, number of neurons in each hidden layer, learning rate, batch size, optimizer, activation,number of training epoches. We identified that the feature engineering(decided by the number of grams and Minimal Frequency), Neurl network structure(decided by number of hidden layers, number of neurons in each hidden layer), batch size, and training epochs are the most important component to train an expressive, generalizable network. As a result, for each feature engineering method of each task (newsgroup and propernames), we would tune hyper-parameter on number of hidden layers, number of neurons in each hidden layer, batch size, and number of training epochs. And based on the accuracy performance on development set, we select the best hyper-parameter combination. Then, using such selected combination, we fine tuned two leftover hyper-parameters(optimizer and activation) to see their influence on MLP's performance on development set accuracy.

### Step1: Tuning feature engineering, Neurl network structure, batch size, and training epochs
#### **Propernames**
* *feature: (1,1)-Gram, Minimal Frequency = 5.*\
Best development set accuracy:  0.6505357760110612\
hidden layer:  1\
hidden neurons:  15\
batch size:  100\
epoch:  350
* *feature:(1,2)-Gram, Minimal Frequency = 5.*\
Best development set accuracy:  0.7614932595921189\
hidden layer:  2\
hidden neurons:  10\
batch size:  500\
epoch:  300
* *feature:(1,3)-Gram, Minimal Frequency = 5.*\
Best development set accuracy:  0.8002073971655721\
hidden layer:  1\
hidden neurons:  10\
batch size:  300\
epoch:  200
* *feature:(1,4)-Gram, Minimal Frequency = 5.*\
Best development set accuracy:  0.8295886622882821\
hidden layer:  1\
hidden neurons:  10\
batch size:  300\
epoch:  200
* *feature:(2,3)-Gram, Minimal Frequency = 5.*\
Best development set accuracy:  0.8026270307639128\
hidden layer:  1\
hidden neurons:  10\
batch size:  500\
epoch:  200

=> The best development set performance comes from the (1,4)-gram, Minimal Frequency = 5 feature extraction with 1 hidden layer, 10 hidden neurons, batch size of 300, training epoch of 200. And the high development set accuracy achieved is  0.8395886622882821.
#### **Newsgroup**
* *feature:(1,2)-Gram, Minimal Frequency = 5.*\
Best development set accuracy: 0.90234202386213\
hidden layer: 1\
hidden neurons: 15\
batch size: 128\
epoch: 11
* *feature:(1,3)-Gram, Minimal Frequency = 5.*\
Best development set accuracy: 0.8996906760936809\
hidden layer: 1\
hidden neurons: 15\
batch size: 128\
epoch: 13
* *feature:(1,4)-Gram, Minimal Frequency = 5.*\
Best development set accuracy: 0.895271763146266\
hidden layer: 1\
hidden neurons: 15\
batch size: 512\
epoch: 25

=> The best development set performance comes from the (1,2)-gram, Minimal Frequency = 5 feature extraction with 1 hidden layer, 15 hidden neurons, batch size of 128, training epoch of 11. And the high development set accuracy achieved is  0.90234202386213.


### Further fine tuning the optimizer and activation function
optimizer collection = ["Adam","AdaGrad","Adadelta"]\
activation collection = ["relu","sigmoid","tanh","leaky relu"]
#### **Propernames**
* Optimizer:  Adam\
Activation:  relu\
development set accuracy:  0.8074662979605945
* Optimizer:  Adam\
Activation:  sigmoid\
development set accuracy:  0.8081576218458347
* Optimizer:  Adam\
Activation:  tanh\
development set accuracy:  0.8005530591081922
* Optimizer:  Adam\
Activation:  leaky relu\
development set accuracy:  0.8022813688212928
* Optimizer:  AdaGrad\
Activation:  relu\
development set accuracy:  0.8361562391980643
* Optimizer:  AdaGrad\
Activation:  sigmoid\
development set accuracy:  0.8434151399930868
* Optimizer:  AdaGrad\
Activation:  tanh\
development set accuracy:  0.8320082958866228
* Optimizer:  AdaGrad\
Activation:  leaky relu\
development set accuracy:  0.8385758727964051
* Optimizer:  Adadelta\
Activation:  relu\
development set accuracy:  0.8340822675423436
* Optimizer:  Adadelta\
Activation:  sigmoid\
development set accuracy:  0.8389215347390252
* Optimizer:  Adadelta\
Activation:  tanh\
development set accuracy:  0.8299343242309022
* Optimizer:  Adadelta\
Activation:  leaky relu\
development set accuracy:  0.8344279294849637

=> Best development set accuracy:  0.8434151399930868\
Optimizer:  AdaGrad\
Activation:  sigmoid

#### **Newsgroup**
* Optimizer:  Adam\
Activation:  relu\
development set accuracy:  0.906319
* Optimizer:  Adam\
Activation:  sigmoid\
development set accuracy:  0.915157
* Optimizer:  Adam\
Activation:  tanh\
development set accuracy:  0.904551
* Optimizer:  Adam\
Activation:  leaky relu\
development set accuracy:  0.911180
* Optimizer:  AdaGrad\
Activation:  relu\
development set accuracy:  0.903226
* Optimizer:  AdaGrad\
Activation:  sigmoid\
development set accuracy:  0.909854
* Optimizer:  AdaGrad\
Activation:  tanh\
development set accuracy:  0.908529
* Optimizer:  AdaGrad\
Activation:  leaky relu\
development set accuracy:  0.904993
* Optimizer:  Adadelta\
Activation:  relu\
development set accuracy:  0.883783
* Optimizer:  Adadelta\
Activation:  sigmoid\
development set accuracy:  0.885550
* Optimizer:  Adadelta\
Activation:  tanh\
development set accuracy:  0.884666
* Optimizer:  Adadelta\
Activation:  leaky relu\
development set accuracy:  0.878038

=> Best development set accuracy:  0.915157\
Optimizer:  Adam\
Activation:  sigmoid
