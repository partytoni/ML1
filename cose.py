import jsonlines
import pandas as pd

TRAIN_PATH = "train_dataset.jsonl"
TEST_PATH = "split_2000.jsonl"
COLUMNS = ['instructions','opt','compiler']

dataframe = pd.DataFrame(columns=COLUMNS)
#instructions, opt, compiler
with jsonlines.open(TEST_PATH) as reader:
    i=0
    for elem in reader:
        dataframe.loc[i] = [elem['instructions'], elem['opt'],elem['compiler']]
        i+=1
        print(i)
    
    dataframe.to_pickle("train-lollo.pkl")

print()

