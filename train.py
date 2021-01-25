import sys
import pandas as pd
import numpy as np
import pickle
import joblib

# for reproducibility of our results
np.random.seed(1234)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#Some helper functions

def get_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)

    INPUT:
    y_true - actual variable
    y_pred - predicted variable

    OUTPUT:
    mape - Mean Absolute Percentage Error (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mape

def get_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE)

    INPUT:
    y_true - actual variable
    y_pred - predicted variable

    OUTPUT:
    rmse - Root Mean Squared Error
    """
    rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))

    return rmse

def load_process_data(data_filepath):
    """
    Load data from filepath, drop dupliccates and sort data by datetime

    INPUT:
    data_filepath - path to data csv file

    OUTPUT:
    df - cleaned and processed dataframe
    """
    df = pd.read_csv(data_filepath, error_bad_lines=False)

    # drop duplicates
    df = df.drop_duplicates()

    # convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

    # sort by datetime
    df.sort_values(by = 'Date', inplace = True, ascending = True)

    return df

def split_data(test_size, df):
    """
    Split data into training and testing sets

    INPUT:
    test_size - size of testing data in ratio (default = 0.2)
    df - cleaned and processed dataframe

    OUTPUT:
    train - training data set
    test - testing data set
    """
    test_size = test_size
    training_size = 1 - test_size

    test_num = int(test_size * len(df))
    train_num = int(training_size * len(df))

    train = df[:train_num][['Date', 'Close']]
    test = df[train_num:][['Date', 'Close']]

    return train, test

def get_x_y(data, N, offset):
    """
    Split data into input variable (X) and output variable (y)

    INPUT:
    data - dataset to be splitted
    N - time frame to be used
    offset - position to start the split

    OUTPUT:
    X - input variable
    y - output variable
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y

def fit_lstm(X_train, y_train, lstm_units = 50, optimizer = 'adam', epochs = 1,
             batch_size = 1, loss = 'mean_squared_error'):
    """
    Create, compile and fit LSTM network

    INPUT:
    X_train - training input variables (X)
    y_train - training output variable (y)

    default(initial) parameters chosen for LSTM
    --------------------------------------------
    lstm_units = 50
    optimizer = 'adam'
    epochs = 1
    batch_size = 1
    loss = 'mean_squared_error'

    OUTPUT:
    model - fitted model
    """
    model = Sequential()
    model.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (X_train.shape[1],1)))
    model.add(LSTM(units = lstm_units))
    model.add(Dense(1))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1)
    # verbose changed to 1 to show the animated progress...
    return model

def get_pred_closing_price(df, test, scaler, model):
    """
    Predict stock price using past 60 stock prices

    INPUT:
    df - dataframe that has been preprocessed
    scaler - instantiated object for MixMaxScaler()
    model - fitted model

    OUTPUT:
    closing_price - predicted closing price using fitted model
    """
    inputs = df['Close'][len(df) - len(test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    return closing_price

def model_performance(test, closing_price):
    """
    Evaluate model performance

    INPUT:
    test - test dataset that contains only 'Date' & 'Close' columns
           (i.e.test = df[train_num:][['Date', 'Close']])
    closing_price - predicted closing price using fitted model

    OUTPUT:
    rmse_lstm - RMSE for LSTM
    mape_lstm - MAPE(%) for LSTM
    """
    test['Predictions_lstm_tuned'] = closing_price
    rmse_lstm = get_rmse(np.array(test['Close']), np.array(test['Predictions_lstm_tuned']))
    mape_lstm = get_mape(np.array(test['Close']), np.array(test['Predictions_lstm_tuned']))
    print('Root Mean Squared Error: ' + str(rmse_lstm))
    print('Mean Absolute Percentage Error (%): ' + str(mape_lstm))
    return rmse_lstm, mape_lstm

def train_pred_eval_model(X_train, y_train, df, scaler, test,
                          lstm_units = 50, optimizer = 'adam', epochs = 1,
                          batch_size = 1, loss = 'mean_squared_error'):
    """
    INPUT:
    X_train - training input variables (X)
    y_train - training output variable (y)
    df - dataframe that has been preprocessed
    scaler - instantiated object for MixMaxScaler()
    test - test dataset that contains only 'Date' & 'Close' columns
           (i.e.test = df[train_num:][['Date', 'Close']])

    default(initial) parameters chosen for LSTM
    --------------------------------------------
    lstm_units = 50
    optimizer = 'adam'
    epochs = 1
    batch_size = 1
    loss = 'mean_squared_error'

    OUTPUT:
    rmse_lstm - RMSE for LSTM
    mape_lstm - MAPE(%) for LSTM
    """
    model_tuned = fit_lstm(X_train, y_train, int(lstm_units), optimizer,
                int(epochs), int(batch_size), loss)
    closing_price_tuned = get_pred_closing_price(df, scaler, model_tuned)
    rmse_lstm, mape_lstm = model_performance(test, closing_price_tuned)
    return rmse_lstm, mape_lstm

def save_model(model, model_filepath):
    """
    INPUT:
    model - trained LSTM model
    model_filepath - location to save the model

    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():

    if len(sys.argv) == 7:
        data_filepath, model_filepath, scaler_filepath, test_size, N, offset = sys.argv[1:]
        print('Loading data...')
        df = load_process_data(data_filepath)
        print(" ")

        print('Splitting data to train and test dataset...')
        train, test = split_data(float(test_size), df)
        print(" ")

        # scale our dataset
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])
        scaled_data_train = scaled_data[:train.shape[0]]

        print('Splitting scaled_data into input (X) and output (y) variables...')
        X_train, y_train = get_x_y(scaled_data_train, int(N), int(offset))
        print(" ")

        print('Create, compile and fit LSTM...')
        model = fit_lstm(X_train, y_train)
        print(" ")

        print('Predict stock price using past 60 stock prices...')
        closing_price = get_pred_closing_price(df, test, scaler, model)
        print(" ")

        print('Evaluate model performance...')
        rmse_lstm, mape_lstm = model_performance(test, closing_price)
        print(" ")

        print('Saving model...')
        save_model(model, model_filepath)
        print(" ")

        print('Saving scaler file...')
        joblib.dump(scaler, scaler_filepath)

    else:
        print('Please provide filepath for the data as the first argument')
        print('Please provide test size in ratio (i.e. 0.2) in second argument')
        print('Please provide time frame for data (provide "60" for 60 days) as third arugment')
        print('Please provide offset (can put "60" for 60 days) as forth argument')


if __name__ == '__main__':
    main()
