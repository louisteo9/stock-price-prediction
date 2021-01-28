# Stock Price Prediction
## Table of Contents
1. [Introduction](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#introduction)
2. [File Descriptions](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#file-descriptions)
3. [Python Libraries Used](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#python-libraries-used)
4. [Instructions](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#instructions)
5. [Acknowledgements](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#acknowledgement)
6. [Screenshots](https://github.com/louisteo9/stock-price-prediction/blob/main/README.md#screenshots)

## Introduction
This project is my capstone project for Udacity's Data Scientist Nanodegree Program.

We will implement two machine learning algorithms (Moving Average and LSTM) to predict future stock price of a company. Then we choose the best out of the two algorithms to develop our own stock price predictor.

This project includes Python scripts where trader can input historical stock price data to get a trained LSTM model. The trained model can then be used to predict future stock price.

Blog post accompanying this project<br/>
https://medium.com/ai-in-plain-english/how-i-create-my-stock-price-predictor-f38c229b758e

## File Descriptions
**Stock Price Prediction.ipynb** - Jupyter Notebook used for stock price prediction.<br/>
**train.py** - Python script to loan historical stock price data and train a LSTM model.<br/>
**predictor.py** - Python script to loan historical stock price data and future predict stock price.

### Folder: data
**EOD-INTC.csv** - historical stock price data for [Intel Inc](https://www.quandl.com/data/EOD/INTC-Intel-Corporation-INTC-Stock-Prices-Dividends-and-Splits) from [Quandl](https://www.quandl.com/).

### Folder: model
**model.pkl** - trained LSTM model in pickle file.<br/>
**scaler.gz** - data scaler object saved in archive file (for use in predictor.py).

## Python Libraries Used
sys, pandas, numpy, sklearn, keras, pickle, joblib, matplotlib

## Instructions
1. Run the following command in the project's root directory to load historial stock price data and output a trained LSTM model and data scaler object file.<br/>
    - python train.py < historical stock price in CSV filepath > < output model filepath in *.pkl* > < output data scaler object filepath in *.gz* > < test size ratio > < training time frame (days) > < offset for training data; if unsure, use training time frame ><br/>
      ` python train.py data/EOD-INTC.csv model/model.pkl model/scaler.gz 0.2 60 60`
2. Run the following command in the project's root directory to load model pickle file, data scaler object file and historical stock price. The predicted price will be shown.<br/>
    - python predictor.py < model filepath > < data scaler object filepath > < historical stock price in CSV filepath ><br/>
      ` python predictor.py model/model.pkl model/scaler.gz data/EOD-INTC.csv`

## Acknowledgement
[Udacity](https://www.udacity.com/) for providing an excellent Data Scientist training program.

## Screenshots
1. Run train.py<br/>
![image](https://github.com/louisteo9/stock-price-prediction/blob/main/screenshots/run%20train%20py.JPG)

2. Run predictor.py<br/>
![image](https://github.com/louisteo9/stock-price-prediction/blob/main/screenshots/run%20predictor%20py.JPG)
