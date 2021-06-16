import pandas as pd
import torch
import numpy as np
import string  
import re, unicodedata
 # set the seed for reproducable result
SEED = 2021
torch.manual_seed(SEED) 


def propername_featurize(input_data,N, MinFreq,model_choice ="NGram"):
    """ Featurizes an input for the proper name domain.

    Inputs:
        input_data: The input data.
    """
    def to_lowercase(text):
        return text.lower()

    def remove_URL(text):
        return re.sub(r"http\S+", "", text)
    def remove_non_ascii(words):
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def tokenize(text):
        return text.split()
    def remove_stopwords(words):
        new_words = []
        for word in words:
            if word not in stop_word:
                new_words.append(word)
        return new_words
    def detokenize_words(words):
        separator = ' '
        return separator.join(words)
    def preprocess_text(df):
        df['text'] = df['text'].apply(to_lowercase)
        df['text'] = df['text'].apply(remove_URL)
        df['text'] = df['text'].apply(tokenize)
        df['text'] = df['text'].apply(remove_non_ascii)
        df['text'] = df['text'].apply(detokenize_words)    
        return df
    def character_ngram(text_matrix, N, MinFreq): #array of non-tokenized text
    #tokenize
        all_tokenized_text = []
        #build all token
        flatten_tokenized_text = []
        for j in text_matrix:
            cur_text = "".join(j.split())
            cur_feature = []
           
            for i in range(N[0]-1,N[1]):   
                
                for l in range(len(cur_text) - i):
                    cur_feature.append(cur_text[l:l+i+1])
               
            all_tokenized_text.append(cur_feature)
            flatten_tokenized_text.extend(cur_feature)
        charfreq = {}
        for i in flatten_tokenized_text:
            if i not in charfreq.keys():
                charfreq[i] = 1
            else:
                charfreq[i] += 1
        selected_feature = []
        for i, item in charfreq.items():
            if item >= MinFreq:
                selected_feature.append(i)
        dim = len(selected_feature)
        encoded_matrix = []
        selected_feature = np.array(selected_feature)
        for i in all_tokenized_text:
            cur_text = np.array(i)
            cur_encoded = np.zeros(dim)
            cur_idx = []
            for j in range(len(cur_text)):
                idx =  np.where(selected_feature == cur_text[j])   
                if len(idx[0]) != 0:        
                    cur_idx.append(idx[0][0])
            #binary character presence 
            cur_encoded[cur_idx] = 1

            encoded_matrix.append(cur_encoded)
        encoded_matrix = np.array(encoded_matrix)

        return encoded_matrix, selected_feature
    def task_specific_featurize(feature_value):
        feature_dic = {"contain_numerics":[], "contain_special_punc":[],"contain_inc":[],"Small_token_length":[]}
        special_pun = "&\?-:%"
        company_col = ["co.","inc."]
        def hasNumbers(string):
            return any(char.isdigit() for char in string)
        for i in text_feature:
            if hasNumbers(i):
                feature_dic["contain_numerics"].append(1)
            else:
                feature_dic["contain_numerics"].append(0)
            Spec_Punc = False
            for l in special_pun:
                if i.find(l) != -1:
                    feature_dic["contain_special_punc"].append(1)
                    Spec_Punc = True
                    break
            if Spec_Punc == False:
                feature_dic["contain_special_punc"].append(0)
            Contain_Com = False
            for l in company_col:
                if i.find(l) != -1:
                    feature_dic["contain_inc"].append(1)
                    Contain_Com = True
                    break
            if Contain_Com == False:
                feature_dic["contain_inc"].append(0)
            token_length = len(i.split())
            if token_length <= 1:
                feature_dic["Small_token_length"].append(1)
            else:
                feature_dic["Small_token_length"].append(0)

        encoded_matrix = pd.DataFrame(feature_dic).values
        selected_feature = list(feature_dic.keys())            
        return encoded_matrix, selected_feature
    # TODO: Implement featurization of input.
    matrix_processed = preprocess_text(input_data)
    text_feature = matrix_processed[["text"]].values.flatten() 
    if model_choice == "NGram":
           
        encoded_matrix, selected_feature = character_ngram(text_feature, N, MinFreq)
    elif model_choice == "TS":
        encoded_matrix, selected_feature = task_specific_featurize(text_feature)
    elif model_choice == "Combined":

        encoded_matrix_specific, selected_feature_specific = task_specific_featurize(text_feature)          
        encoded_matrix_bow, selected_feature_bow = character_ngram(text_feature, N, MinFreq)
        encoded_matrix = np.hstack((encoded_matrix_bow,encoded_matrix_specific))
        selected_feature = list(selected_feature_bow)
        selected_feature.extend(selected_feature_specific)
        
    return encoded_matrix,selected_feature
   

def propername_data_loader(train_data_filename,
                           train_labels_filename,
                           dev_data_filename,
                           dev_labels_filename,
                           test_data_filename,N, MinFreq, model_choice):

    train_x = pd.read_csv(train_data_filename)[["text"]].values
    dev_x = pd.read_csv(dev_data_filename)[["text"]].values
    test_x = pd.read_csv(test_data_filename)[["text"]].values
    train_y = pd.read_csv(train_labels_filename)[["type"]].values
    dev_y = pd.read_csv(dev_labels_filename)[["type"]].values
    labels = np.unique(train_y)
    test_y = np.random.randint(low= 0, high = 20, size=(len(test_x),1))
    df_train = pd.DataFrame({'text': train_x.flatten(), 'label': train_y.flatten()})
    df_dev = pd.DataFrame({'text': dev_x.flatten(), 'label': dev_y.flatten()})
    df_test = pd.DataFrame({'text': test_x.flatten(), 'label': test_y.flatten()})
    train_length = len(df_train)
    dev_length = len(df_dev)
    df_data = pd.concat([df_train, df_dev, df_test])
    #encoded on character-level matrix
    encoded_matrix,selected_feature = propername_featurize(df_data,N, MinFreq,model_choice)
    train_x_featurized = encoded_matrix[:train_length]
    dev_x_featurized = encoded_matrix[train_length:train_length+dev_length]
    test_x_featurized = encoded_matrix[train_length+dev_length:]
    #insert the label to the last position of the matrix
    train_featurized = np.hstack((train_x_featurized, train_y))
    dev_featurized = np.hstack((dev_x_featurized, dev_y))
    test_featurized = np.hstack((test_x_featurized, test_y))
    #return 2 dimensional matrix with last column being label
    return train_featurized, dev_featurized, test_featurized

"""
    Data Process + Feature Engineering
    Usage: train_featurized, dev_featurized, test_featurized =  propername_data_loader("data/propernames/train/train_data.csv",
                         "data/propernames/train/train_labels.csv",
                          "data/propernames/dev/dev_data.csv",
                          "data/propernames/dev/dev_labels.csv",
                          "data/propernames/test/test_data.csv",N range in Character N-Gram, MinFreq, Feature extractor)
    
    Input explaination:
    N range in BOW: a tupe (M, N) specifies the range of N to extract in the N-Gram
    MinFreq: Minimal frequency of a character / sequence to be taken as a feature, for dimension control purpose
    Feature extractor: which approach to take for feature extraction, takes the following three values: 
            "NGram" : Character-level N-Gram
            "TS" : Task Specific model (4 features for now). => by observation of difference
            "Combined": Combination of "NGram" and "TS"
"""
