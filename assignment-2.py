import torch
import torch.nn as nn
import yfinance
import datetime

torch.manual_seed(7373) # seed for reproducability



class YahooLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) # main LSTM layer
        self.linear = nn.Linear(hidden_size, 1) # final output layer

    def forward(self, x): # forward pass definition
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def load_data(ticker, start, end):
    df = yfinance.download(ticker, start=start, end=end) # import data
    df.dropna(inplace=True) # drop null values
    return df['Close'].sort_index() # return datafram wtih sorted index

def split_data(data, tstart, tend):
    test_start = str(int(tend) + 1) # determine start for test set
    train_set = data.loc[tstart:tend] # split training set
    test_set = data.loc[test_start:] # split test set
    return train_set, test_set # return both train and test set

def create_timeseries_windows(data, window_length):
    
    X, y = [], [] # initialize arrays
    for i in range(len(data) - window_length):
        feature = data[i: i + window_length] # split from i to window length
        target = data[i+1: i + window_length + 1] # split out target values
        X.append(feature) # collect feature
        y.append(target) # collect target

    return torch.Tensor(X), torch.Tensor(y) # return as tensors



ticker = 'AAPL'
load_start = '2015-01-01'
load_end = datetime.datetime.now()
split_start = '2019'
split_end = '2023'
window_length = 3
batch_size = 32



stock_data = load_data(ticker=ticker, start=load_start, end=load_end) # get stock data
train, test = split_data(data=stock_data, tstart=split_start, tend=split_end) # split stock data into train test sets

X_train, y_train = create_timeseries_windows(data=train.values, window_length=window_length) # create time series training sets
X_test, y_test = create_timeseries_windows(data=test.values, window_length=window_length) # create time series test sets



model = YahooLSTM()
optimizer = torch.optim.Adam(params=model.parameters())
loss_function = nn.MSELoss()
data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
epochs = 300



for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in data_loader:
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 50 != 49:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = torch.sqrt(loss_function(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = torch.sqrt(loss_function(y_pred, y_test))
        
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

torch.save(model.state_dict(), "model.pth")