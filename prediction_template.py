import pandas as pd

template='Model Run Date,Actual Close Price On Model Run Date,Prediction,Stock,Buy,Sell,Hold'.split(',')
predictions = pd.DataFrame(columns=template)
predictions.to_csv('predictions.csv')
