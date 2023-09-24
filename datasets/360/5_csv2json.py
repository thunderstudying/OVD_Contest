import json


phase = 'test'

with open(f'4_{phase}_cat_info_new.csv', 'r', encoding='utf-8-sig') as file:
# with open(f'4_{phase}_cat_info.csv', 'r', encoding='utf-8-sig') as file:
# with open(f'new_test_info.csv', 'r', encoding='utf-8-sig') as file:
    lis = []
    for line in file:
        line = line.replace('\n', '')
        lis.append(line.split(','))

for i in range(1, len(lis)):
    # for ind, item in enumerate(lis[i]):
    #     if ind == 0 or ind == 4 or ind == 5:
    #         lis[i][ind] = eval(item)
    lis[i] = dict(zip(lis[0], lis[i]))

json.dump(lis[1:], open(f'5_{phase}_cat_info.json', 'w'))
# json.dump(lis[1:], open(f'5_tt.json', 'w'))
