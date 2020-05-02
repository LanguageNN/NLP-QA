#from nlp hw5

# Make sure that execution of this cell doesn't return any errors. If it does, go the class repository and follow the environment setup instructions
import random
import itertools
from collections import Counter, defaultdict
import string
import json

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.style.use('seaborn')



def plot_label_counts(train, test):
    ### YOUR CODE BELOW ###
    counter_train = Counter(train)
    counter_test = Counter(test)
    ### YOUR CODE ABOVE ###

    names = [c for c, n in counter_train.most_common()]
    index = np.arange(len(names))

    values_train = [counter_train[c] for c in names]
    values_test = [counter_test[c] for c in names]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(index-0.2, values_train, label="train", width=0.4)
    ax.bar(index+0.2, values_test, label="test", width=0.4)

    ax.set_xticks(index)
    ax.set_xticklabels(names, rotation=30)

    ax.legend()
    fig.tight_layout()
    plt.show()


def transform_to_tfidf(train_corpus, test_corpus, max_df=1.0, min_df=1, stop_words=None):
    """
    Transform train and test documents to their TF-IDF representations and return the features.
    Args: 
        train_corpus (list) : a list of str documents
        test_corpus (list) : a list of str documents
    Returns:
        a tuple of the transformed train matrix, the transformed test matrix and the list of words used by the TfidfVectorizer as features
    
    """
    ### YOUR CODE BELOW ###
    train_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    X_train = train_vectorizer.fit_transform(train_corpus)
    X_test = train_vectorizer.transform(test_corpus)
    feature_names = train_vectorizer.get_feature_names()
    ### YOUR CODE ABOVE ###
    return X_train.toarray(), X_test.toarray(), feature_names


class MyVocabulary:
    def __init__(self, special_tokens=None):
        self.w2idx = {}
        self.idx2w = {}
        self.w2cnt = defaultdict(int)
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.add_tokens(special_tokens)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.w2cnt[token] += 1

    def add_token(self, token):
        if token not in self.w2idx:
            cur_len = len(self)
            self.w2idx[token] = cur_len
            self.idx2w[cur_len] = token

    def prune(self, min_cnt=2):
        to_remove = set([token for token in self.w2idx if self.w2cnt[token] < min_cnt])
        to_remove ^= set(self.special_tokens)

        for token in to_remove:
            self.w2cnt.pop(token)

        self.w2idx = {token: idx for idx, token in enumerate(self.w2cnt.keys())}
        self.idx2w = {idx: token for token, idx in self.w2idx.items()}

    def __contains__(self, item):
        return item in self.w2idx

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.w2idx[item]
        elif isinstance(item , int):
            return self.idx2w[item]
        else:
            raise TypeError("Supported indices are int and str")

    def __len__(self):
        return(len(self.w2idx))

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab=None, labels_vocab=None, max_len=40, lowercase=True):
        """
        Args:
            texts (list of str): texts of the dataset examples
            labels (list of str): the correponding labels of the dataset examples
            vocab (MyVocabulary, optional): vocabular to convert text to indices. If not provided, will be created based on the texts
            labels_vocab (MyVocabulary, optional): vocabular to convert labels to indices. If not provided, will be created based on the labels
            max_len (int): maximum length of the text. Texts shorter than max_len will be cut at the end
            lowercase (bool, optional): a flag specifying whether or not the input text should be lowercased
        """
        
        self.max_len = max_len
        self.lowercase = lowercase

        self.texts = [self._preprocess(t, max_len=max_len, lowercase=lowercase) for t in texts]
        self.labels = labels

        if vocab is None:
            vocab = MyVocabulary(['<PAD>', '<UNK>'])
            vocab.add_tokens(itertools.chain.from_iterable(self.texts))

        if labels_vocab is None:
            labels_vocab = MyVocabulary()
            labels_vocab.add_tokens(labels)
            
        self.vocab = vocab
        self.labels_vocab = labels_vocab

    def _preprocess(self, text, max_len=None, lowercase=True):
        """
        Preprocess a give dataset example
        Args:
            text (str): given dataset example
            max_len (int, optional): maximum sequence length
            lowercase (bool, optional): a flag specifying whether or not the input text should be lowercased
        
        Returns:
            a list of tokens for a given text span
        """

        # tokenize the input text
        ### YOUR CODE BELOW ###
        if lowercase:
            text = text.lower()
        tokens = word_tokenize(text)
        
        # cut the list of tokens to `max_len` if needed 
        tokens = tokens[:max_len]
        ### YOUR CODE ABOVE ###

        return tokens

    def _pad(self, tokens):
        """
        Pad tokens to self.max_len
        Args:
            tokens (list): a list of str tokens for a given example
            
        Returns:
            list: a padded list of str tokens for a given example
        """
        # pad the list of tokens to be exactly of the `max_len` size
        ### YOUR CODE BELOW ###
        while len(tokens) < self.max_len:
            tokens.append(self.vocab.special_tokens[0])
        ### YOUR CODE ABOVE ###
        return tokens

    def __getitem__(self, idx):
        """
        Given an index, return a formatted dataset example
        
        Args:
            idx (int): dataset index
            
        Returns:
            tuple: a tuple of token_ids based on the vocabulary mapping  and a corresponding label
        """
        ### YOUR CODE BELOW ###
        tokens = []
        for token in self.texts[idx]:
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                tokens.append(self.vocab['<UNK>'])
        tokens = tuple(tokens)
        label = self.labels_vocab[self.labels[idx]]
        ### YOUR CODE ABOVE ###
        
        return tokens, label

    def __len__(self):
        return len(self.texts)


class TextClassificationModel(torch.nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, nb_classes):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nb_classes = nb_classes

        ### YOUR CODE BELOW ###
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, nb_classes)
        ### YOUR CODE ABOVE ###
        
        
    def forward(self, inputs):
        ### YOUR CODE BELOW ###
        embvec = self.embedding(inputs[0]).unsqueeze(0)
        output, hn = self.gru(embvec)
        maxpool = torch.max(output, 0)
        logits = self.linear(maxpool.values)
        ### YOUR CODE ABOVE ###        
        
        return logits

class DataClassifier:
    def __init__(self, data, labels):
        data_train, data_test, labels_train, labels_test = train_test_split(self.data, self.labels, test_size=0.3, random_state=10)

        self.dataset_train = TextClassificationDataset(data_train, labels_train)
        self.dataset_test = TextClassificationDataset(data_test, labels_test, vocab=dataset_train.vocab, labels_vocab=dataset_train.labels_vocab)

        # DATALOADER #
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=64)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=64)

        # MODEL INITIALIZATION #
        self.model = TextClassificationModel(64, len(dataset_train.vocab), 128, len(needed_categories))

        # OPTIMIZER #
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # LOSS-FUNCTION #
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train_data(self):
        # TRAINING #
        losses = []
        for epoch in range(10):
            epoch_losses = []
            for i, batch in enumerate(self.dataloader_train):
                self.optimizer.zero_grad()
                
                x, y = batch
                
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                epoch_losses.append(loss.item())
                
                loss.backward()
                self.optimizer.step()

            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)
            print('Epoch {}, loss {}'.format(epoch, epoch_loss))
        
        return losses

    def predict(self, dataloader):
        """
        Predict probability distributions over classes on the test data
        
        Args:
            model: your torch.nn.Module() model object
            dataloader: test Dataloder() object
            
        Returns:
            tuple: np.array/list with true labels and np.array/list with predicted labels
        """
        y_pred = []
        y_true = []
        
        ### YOUR CODE BELOW ###
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                logits = self.model(x)
                y_pred.append(torch.argmax(logits, 1).numpy())
                y_true.append(y.numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        ### YOUR CODE ABOVE ###
            
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)    
        
        return y_true, y_pred

    def accuracy_score(self):
        y_true_train, y_pred_train = self.predict(self.dataloader_train)
        y_true_test, y_pred_test = self.predict(self.dataloader_test)

        ### YOUR CODE BELOW ###
        acc_train = accuracy_score(y_true_train, y_pred_train)
        acc_test = accuracy_score(y_true_test, y_pred_test)
        ### YOUR CODE ABOVE ###

        print('Accuracy train', acc_train)
        print('Accuracy test', acc_test)























