import pandas as pd
from sklearn.metrics import accuracy_score

FILEPATH1 = "results.csv"
FILEPATH2 = "lorix.csv"

uno = []
due = []
with open(FILEPATH1,'r') as file1:
    with open(FILEPATH2, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        error=0
        for i in range(len(lines1)):
            line1 = lines1[i]
            line2 = lines2[i]
            uno.append(line1)
            due.append(line2)
            if line1 != line2:
                print(line1, "diverso da", line2)
                error+=1
print("accuracy:",accuracy_score(uno,due))
print("num of errors:",error)




