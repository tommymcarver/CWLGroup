import pandas as pd

df = pd.read_csv('data/1950-2019/bollywood_full_1950-2019.csv')
f = open('plots.txt', 'x')
count = 1
for story in df['story']:
    if type(story).__name__ == 'str':
        f.write(str(count) + "\n\n")
        f.write(story + "\n\n")
        count+=1
    if count == 1000:
        break
