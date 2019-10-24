import pandas as pd

TRAIN_PATH = "train.pkl"
TEST_PATH = "test.pkl"

dataframe = pd.read_pickle(TRAIN_PATH)

instructions_set = set()
instructions_column = dataframe['instructions'].tolist()
for elem in instructions_column:
    for instruction in elem:
        first = instruction.split(" ")[0]
        instructions_set.add(first)

# instructions contiene il set di tutte le istruzioni trovate in tutte le righe (no ripetizioni)
instructions = list(instructions_set)
dict=dict()
i=0
for elem in instructions:
   dict[elem]=i
   i+=1
    
df = pd.DataFrame(columns=instructions)

counter = 0
#elem is a list of instructions
for elem in instructions_column:

    row = [0] * len(instructions)

    for instruction in elem:
        first = instruction.split(" ")[0]
        row[dict[first]] += 1

    df.loc[counter] = row
    counter += 1

    print(counter)

df.to_pickle('matrix.pkl')

