import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from gensim.utils import simple_preprocess
import pickle


class sentence_dataset():
    def __init__(self, docs_series, block_count=10):
        self.block_count = block_count
        self.docs_series = docs_series

    def get_sentences(self):
        sentences = []
        for i, doc in enumerate(self.docs_series):
            splitted = simple_preprocess(doc)
            length = len(splitted) // self.block_count
            sent_vec = []
            if self.block_count == 1:
                sentences.append([' '.join(splitted)])
            else:
                for j in range(self.block_count - 1):
                    sent_vec.append(' '.join(splitted[j*length: (j + 1)*length]))
                sent_vec.append(' '.join(splitted[(self.block_count - 1) * length: -1]))
                sentences.append(sent_vec)
        return sentences

class finBertsentiment():
    def __init__(self):
        if os.path.exists() and os.path.exists():
            with open('finBertML.model', 'rb') as f:
                finbert = pickle.load(f)
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.finbert = finbert.to(self.device)

            with open('tokenizerML.model', 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:

            finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

            with open('finBertML.model', 'wb') as f:
                pickle.dump(finbert, f)

            with open('tokenizerML.model', 'wb') as f:
                pickle.dump(tokenizer, f)

    def get_sentiment(self, docs):
        #sentenses = ['The company has shown high profitability in the current quarter',
        #             'Another firm suffered serious losses in the new year',
        #             'The company has had stable growth this year']
        inputs = self.tokenizer(docs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.finbert(**inputs)[0]
        out = outputs.detach().cpu().numpy().T
        return outputs, [np.mean(o) for o in out]


def main():
    news_dataset = pd.read_csv('dataset/GT_news_dataset.csv')

    news_sent = sentence_dataset(news_dataset['content'], block_count=10)
    content = news_sent.get_sentences()

    news_sent = sentence_dataset(news_dataset['title'], block_count=1)
    title = news_sent.get_sentences()

    finBert = finBertsentiment()

    out_ds = {'ticker': [],
              'title': [],
              'category': [],
              'date': [],
              'provider': [],
              'content_neu': [],
              'content_pos': [],
              'content_neg': [],
              'title_neu': [],
              'title_pos': [],
              'title_neg': []
              }
    for i, doc in enumerate(tqdm(content)):
        try:
            sentiment, mean_sent = finBert.get_sentiment(doc)
            sentiment, mean_sent_t = finBert.get_sentiment(title[i])

            out_ds['ticker'].append(news_dataset['ticker'].loc[i])
            out_ds['title'].append(news_dataset['title'].loc[i])
            out_ds['category'].append(news_dataset['category'].loc[i])
            out_ds['date'].append(news_dataset['date'].loc[i])
            out_ds['provider'].append(news_dataset['provider'].loc[i])
            out_ds['content_neu'].append(mean_sent[0])
            out_ds['content_pos'].append(mean_sent[1])
            out_ds['content_neg'].append(mean_sent[2])
            out_ds['title_neu'].append(mean_sent_t[0])
            out_ds['title_pos'].append(mean_sent_t[1])
            out_ds['title_neg'].append(mean_sent_t[2])
        except:
            print(f"Error: {i} lenght: {len(news_dataset['content'].loc[i])}")

    out_DF = pd.DataFrame(out_ds)
    print(out_DF)
    out_DF.to_csv('dataset/finbert_dataset.csv')



if __name__=='__main__':
    main()