import json


phase = 'test'
file = json.load(open(f'3_{phase}_cat_info.json', 'r'))
title = file[0].keys()
output = ','.join([*title])
for item in file:
    output += '\n'
    for key in title:
        value = f'{item[key]}'
        output += value + ',' if key != list(title)[-1] else value
with open(f'4_{phase}_cat_info.csv', 'w', encoding='utf-8-sig') as f:
    f.write(output)
