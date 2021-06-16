
import torch
import numpy as np
import string
import re, unicodedata
SEED = 2021
torch.manual_seed(SEED) 
 # source for english stopwords: https://gist.github.com/sebleier/554280
stop_word = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

def newsgroup_featurize(input_data,N, MinFreq, model_choice ="BOW"):#, N, MinDf
    """ Featurizes an input for the newsgroup domain.

    Inputs:
        input_data: The input data, DataFrame
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
    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]','', word) 
            if new_word != '':
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
        df['text'] = df['text'].apply(remove_stopwords)
        df['text'] = df['text'].apply(remove_non_ascii)
        df["text"] = df['text'].apply(remove_punctuation)
        df['text'] = df['text'].apply(detokenize_words)
        
        return df
    def bag_of_words(text_matrix, N, MinFreq): 
        all_tokenized_text = []
        #build all token
        flatten_tokenized_text = []
        for j in text_matrix:
            cur_feature = []
            tokenized_text = j.split()
            for i in range(N[0]-1,N[1]):   
                if i != 0:
                    for l in range(len(tokenized_text) - i):
                        cur_feature.append(" ".join(tokenized_text[l:l+i+1]))
                else:
                    cur_feature.extend(tokenized_text)
            all_tokenized_text.append(cur_feature)
            flatten_tokenized_text.extend(cur_feature)
        wordfreq = {}
        for i in flatten_tokenized_text:
            if i not in wordfreq.keys():
                wordfreq[i] = 1
            else:
                wordfreq[i] += 1
        selected_feature = []
        for i, item in wordfreq.items():
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
            cur_encoded[cur_idx] = 1

            encoded_matrix.append(cur_encoded)
        encoded_matrix = np.array(encoded_matrix)

        return encoded_matrix, selected_feature
    def task_specific_featurize(feature_value):
        feature_dic = {"host_count":[], "distribution_count":[],"from_university":[],"reference_count":[]}
        host_key = "nntppostinghost"
        distribution_key = "distribution"
        
        for i in feature_value:
            cur_token = i.split()
            feature_dic["host_count"].append(cur_token.count(host_key))
            feature_dic["distribution_count"].append(cur_token.count(distribution_key))
            feature_dic["reference_count"].append(cur_token.count("article"))
            if cur_token[1][-3:] == "edu":
                feature_dic["from_university"].append(1)
            else:
                feature_dic["from_university"].append(0)
        encoded_matrix = pd.DataFrame(feature_dic).values
        selected_feature = list(feature_dic.keys())
        return encoded_matrix, selected_feature

    matrix_processed = preprocess_text(input_data)
    if model_choice == "BOW":
        text_feature = matrix_processed[["text"]].values.flatten()    
        encoded_matrix, selected_feature = bag_of_words(text_feature, N, MinFreq)
    elif model_choice == "TS":
        text_feature = matrix_processed[["text"]].values.flatten()
        encoded_matrix, selected_feature = task_specific_featurize(text_feature)
    elif model_choice == "Combined":
        text_feature = matrix_processed[["text"]].values.flatten()
        encoded_matrix_specific, selected_feature_specific = task_specific_featurize(text_feature)          
        encoded_matrix_bow, selected_feature_bow = bag_of_words(text_feature, N, MinFreq)
        encoded_matrix = np.hstack((encoded_matrix_bow,encoded_matrix_specific))
        selected_feature = list(selected_feature_bow)
        selected_feature.extend(selected_feature_specific)
    
        

    return encoded_matrix, selected_feature

def newsgroup_data_loader(train_data_filename,
                          train_labels_filename,
                          dev_data_filename,
                          dev_labels_filename,
                          test_data_filename,
                          N, MinFreq, model_choice):
    train_x = pd.read_csv(train_data_filename)[["text"]].values
    dev_x = pd.read_csv(dev_data_filename)[["text"]].values
    test_x = pd.read_csv(test_data_filename)[["text"]].values
    train_y = pd.read_csv(train_labels_filename)[["newsgroup"]].values
    dev_y = pd.read_csv(dev_labels_filename)[["newsgroup"]].values
    labels = np.unique(train_y)
    test_y = np.random.randint(low= 0, high = 20, size=(len(test_x),1))

    df_train = pd.DataFrame({'text': train_x.flatten(), 'label': train_y.flatten()})
    df_dev = pd.DataFrame({'text': dev_x.flatten(), 'label': dev_y.flatten()})
    df_test = pd.DataFrame({'text': test_x.flatten(), 'label': test_y.flatten()})
    train_length = len(df_train)
    dev_length = len(df_dev)
    df_data = pd.concat([df_train, df_dev, df_test])
    encoded_matrix, selected_feature = newsgroup_featurize(df_data, N, MinFreq, model_choice)
   
    
    train_x_featurized = encoded_matrix[:train_length]
    dev_x_featurized = encoded_matrix[train_length:train_length+dev_length]
    test_x_featurized = encoded_matrix[train_length+dev_length:]
    train_featurized = np.hstack((train_x_featurized, train_y))
    dev_featurized = np.hstack((dev_x_featurized, dev_y))
    test_featurized = np.hstack((test_x_featurized, test_y))
    return train_featurized, dev_featurized, test_featurized
"""
    Data Process + Feature Engineering
    Usage: train_featurized, dev_featurized, test_featurized= newsgroup_data_loader("data/newsgroups/train/train_data.csv",
                         "data/newsgroups/train/train_labels.csv",
                          "data/newsgroups/dev/dev_data.csv",
                          "data/newsgroups/dev/dev_labels.csv",
                          "data/newsgroups/test/test_data.csv", N range in BOW, MinFreq, Feature extractor)
    
    Input explaination:
    N range in BOW: a tupe (M, N) specifies the range of N to extract in the N-Gram
    MinFreq: Minimal frequency of a word / sequence to be taken as a feature, for dimension control purpose
    Feature extractor: which approach to take for feature extraction, takes the following three values: 
            "BOW" : Bag of Word model
            "TS" : Task Specific model (4 features for now). Reference: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
            "Combined": Combination of "BOW" and "TS"
"""
