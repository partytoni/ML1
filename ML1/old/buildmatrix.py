import pandas as pd

# the train and test pickles are the dataframe transformation from the jsonl file.
# they represent the same objects
TRAIN_PATH = "train.pkl"
TEST_PATH = "test.pkl"

# dataframe = pd.read_pickle(TRAIN_PATH)
dataframe = pd.read_pickle(TEST_PATH)

instructions_set = set()
instructions_column = dataframe['instructions'].tolist()

# we only take the first word of each instructions (we strip eventual parameters)
for elem in instructions_column:
    for instruction in elem:
        first = instruction.split(" ")[0]
        instructions_set.add(first)

# instructions contains the set of all the instruction names found in all the rows (no repetition)
instructions = list(instructions_set)
dictionary = dict()
i = 0

# we build a reverse index with key 'instruction-name' and value an incrementing integer to handle positions
# in the matrix
for elem in instructions:
    dictionary[elem] = i
    i += 1

# we initialize an empty dataframe to be filled with instructions (in case of training file we have
# [instructions, opt, compiler] columns)
df = pd.DataFrame(columns=instructions)
counter = 0
# elem is a list of instructions
for elem in instructions_column:

    # initializing an array of zeroes with length len(instructions)
    row = [0] * len(instructions)

    # we get the instruction name and increment the counter value for that instruction
    # 'row' basically counts the occurrence of any given instruction name
    for instruction in elem:
        first = instruction.split(" ")[0]
        row[dictionary[first]] += 1

    # we insert the row in the dataframe
    df.loc[counter] = row
    counter += 1

# we export the matrix in a pickle.
df.to_pickle('matrix.pkl')
