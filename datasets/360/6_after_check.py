import json
from pprint import pprint

phase = 'test'

ori = json.load(open(f'3_{phase}_cat_info.json'))
new = json.load(open(f'5_{phase}_cat_info.json'))

lis = []
for ind, it in enumerate(new):
    if new[ind] != ori[ind]:
        lis.append(ind)

for item in lis:
    # print(item)
    pprint([f'0-indexed:{item}', ori[item], new[item]])
