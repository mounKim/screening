import torch
import torch.utils.data as data
from embedding import *
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords


class Dataset(data.Dataset):
    def __init__(self, args, data_list):
        super(Dataset, self).__init__()
        self.args = args
        self.data_list = data_list
        self.embedding = return_embedding()
        self.words = return_top10000()[0]
        self.stopword = stopwords.words('english')

    def __getitem__(self, index):
        text = self.data_list[index][0].split()
        label = self.data_list[index][1]
        emb_text = []
        for i in text:
            if i in self.words and i in self.embedding.keys() and i not in self.stopword:
                if self.args.use_embedding:
                    emb_text.append(self.embedding[i])
                else:
                    emb_text.append(self.words.index(i))
        emb_text = pad_sequences([emb_text], maxlen=self.args.maxlen,
                                 dtype='float', truncating='post', padding='post')
        return torch.tensor(emb_text, dtype=torch.float32), torch.tensor(label)

    def __len__(self):
        return len(self.data_list)
