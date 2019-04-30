import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


class StockHMM(object):
    def __init__(self, file, train_size=0.8, hidden_states=4,
                 change_steps=30, high_steps=15, low_steps=15,
                 delta = 0.15, latency=10, type = 0):

        ### variables ###
        self.model = GaussianHMM(n_components=hidden_states)
        self.file = file
        self.name = self.file[:-12]+str(type)
        self.latency = latency
        self.delta = delta
        self.counter = 0

        ### Methods ###
        self.split_data(train_size)
        self.get_outcomes(change_steps, high_steps, low_steps)

    def split_data(self, train_size):
        feats = []
        opens = []
        closes = []
        with open(self.file,'r') as f:
            for line in f:
                tmp = line.rstrip('\n').split(',')
                opens.append(tmp[0])
                closes.append(tmp[1])
                feats.append(tmp[2:])

        opens = np.asarray(opens, dtype=np.float32)
        closes = np.asarray(closes, dtype=np.float32)
        feats = np.asarray(feats, dtype=np.float32)
        data_len = feats.shape[0]
        test_index = int(round(train_size*data_len))

        self.train_data = feats[0:test_index]
        self.test_data = feats[test_index:]
        #print(self.test_data)
        self.opens = opens[test_index:]
        self.closes = closes[test_index:]
        self.dates = np.arange(data_len - test_index)
        #print(self.dates)

    def train(self):
        self.model.fit(self.train_data)

    def get_outcomes(self, change_steps, high_steps, low_steps):
        change_range = np.linspace(-self.delta, self.delta, change_steps)
        high_range = np.linspace(0, self.delta, high_steps)
        low_range = np.linspace(0, self.delta, low_steps)

        self.outcomes = np.array([x for x in itertools.product(change_range, high_range, low_range)])

    def get_probable_outcome(self, date):
        start = max(0, date - self.latency)
        end = max(0, date - 1)
        if start == end:
            previous_observ = self.test_data[start]
        else:
            previous_observ = self.test_data[start: end]

        scores = []
        for possible_outcome in self.outcomes:
            final_observ = np.row_stack((previous_observ, possible_outcome))
            scores.append(self.model.score(final_observ))
        probable_outcome = self.outcomes[np.argmax(scores)]

        return probable_outcome

    def predict_close(self, date):
        self.counter += 1
        print(self.counter)
        open = self.opens[date]
        pred_change, pred_high, pred_low = self.get_probable_outcome(date)
        close = open * (1 + pred_change)
        return close

    def predict_close_series(self):

        predicted_closes = []
        for date in self.dates:
            predicted_closes.append(self.predict_close(date))
        self.predicted_closes = np.asarray(predicted_closes)

    def plot_closes(self):
        print('dates:', self.dates.shape[0])
        print('closes row:',self.closes.shape[0])
        #print('closes col:',self.closes.shape[1])
        print('pred closes row:',self.predicted_closes.shape[0])
        #print('pred closes col:',self.predicted_closes.shape[1])

        df=pd.DataFrame({'x': self.dates, 'actual': self.closes, 'predicted': self.predicted_closes})
        plt.plot( 'x', 'actual', data=df, marker='+', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=2)
        plt.plot( 'x', 'predicted', data=df, marker='+', color='red', markersize=7, linewidth=2)
        plt.legend()
        #plt.show()
        figfile = 'C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/StockPlots/'+self.name+'.pdf'
        plt.savefig(figfile, bbox_inches='tight')

    def save_closes(self):
        closes = np.column_stack((self.closes, self.predicted_closes))
        pd.DataFrame(closes).to_csv('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/Predictions/'+self.name+'.csv')

if __name__ == "__main__":
    files = os.listdir('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/TestStocks')
    os.chdir('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/TestStocks')
    file_len = len(files)
    for j in range(5):
        for i in range(file_len):
            file = files[i]

            if j == 0:
                stock_model = StockHMM(file=file, type = j)
            if j == 1:
                stock_model = StockHMM(file=file, hidden_states = 6, type = j)
            if j == 2:
                stock_model = StockHMM(file=file, change_steps=50, high_steps=20, low_steps=20, type = j)
            if j == 3:
                stock_model = StockHMM(file=file, delta = 0.3, type = j)
            if j == 4:
                stock_model = StockHMM(file=file, latency = 20, type = j)

            stock_model.train()
            stock_model.predict_close_series()
            stock_model.plot_closes()
            stock_model.save_closes()
    # os.chdir('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/NewStocks/')
    # stock_model = StockHMM(file='ablx.us_proc.txt')
    # stock_model.train()
    # stock_model.predict_close_series()
    # stock_model.plot_closes()
    # stock_model.save_closes()
