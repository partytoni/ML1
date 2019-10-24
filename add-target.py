import pandas as pd
dataframe = pd.read_pickle('matrix-lollo.pkl')
train = pd.read_pickle('train-lollo.pkl')
train = train.drop('instructions', axis=1)
dataframe = pd.concat([dataframe,train],axis=1)
dataframe.to_pickle('matrix2-lollo.pkl')
