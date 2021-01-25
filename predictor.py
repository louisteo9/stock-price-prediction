import sys
import numpy as np
import pandas as pd
import pickle
import joblib

# import function defined earlier in train.py
from train import load_process_data

# slightly modified from get_pred_closing_price() from train.py
def pred_closing_price(df, scaler, model):
    """
    Predict stock price using past 60 stock prices

    INPUT:
    df - dataframe that has been preprocessed
    scaler - instantiated object for MixMaxScaler()
    model - fitted model

    OUTPUT:
    predicted_price - predicted closing price
    """
    inputs = df['Close'][len(df) - 278 - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test.append(inputs[-60:,0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    predicted_price = float(closing_price[-1])
    return predicted_price

def main():
    if len(sys.argv) == 4:
        model_filepath, scaler_filepath, data_filepath =sys.argv[1:]

        # load pkl model file
        print('Loading model...')
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
        print(" ")

        print('Loading data...')
        df = load_process_data(data_filepath)
        print(" ")

        print('Loading scaler file...')
        my_scaler = joblib.load(scaler_filepath)
        print(" ")

        print('Predicting closing stock price...')
        predicted_price = pred_closing_price(df, my_scaler, model)
        print(" ")

        print('Predicted price: '+'$ '+str("{:.2f}".format(predicted_price)))


if __name__ == '__main__':
    main()
