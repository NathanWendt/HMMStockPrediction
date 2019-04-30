import os
import numpy as np

files = os.listdir('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/Stocks')
num_files = len(files)
os.chdir('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/Stocks')

for i in range(num_files):
    if os.stat(files[i]).st_size == 0:
        continue
    filename = files[i][:-4]+'_proc'
    print(filename)
    data = []
    with open(files[i],'r') as f:
        for line in f:
            tmp = line.rstrip('\n').split(',')
            data.append(tmp)

    data = data[1:]
    data = [data[n][1:5] for n in range(len(data))]
    data = np.asarray(data, dtype=np.float32)
    data = data[np.all(data != 0, axis=1)]
    data_len = data.shape[0]
    new_data = np.zeros((data_len,5))
    new_data[:,0] = [data[n][0] for n in range(len(data))] #open
    new_data[:,1] = [data[n][1] for n in range(len(data))] #close
    new_data[:,2] = [np.subtract(data[n][3],data[n][0])/data[n][0] for n in range(data_len)] # (close - open)/open
    new_data[:,3] = [np.subtract(data[n][1],data[n][0])/data[n][0] for n in range(data_len)] # (high - open)/open
    new_data[:,4] = [np.subtract(data[n][0],data[n][2])/data[n][0] for n in range(data_len)] # (open - low)/open
    np.savetxt('C:/Users/natha/OneDrive/Documents/WSU/CptS_577/StockMarket/Data/NewStocks/'+filename+'.txt',new_data,delimiter=',')
