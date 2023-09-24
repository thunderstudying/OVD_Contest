import json
import numpy as np
from translate import Translator

# translator = Translator(from_lang='zh', to_lang='en')
#
# train = json.load(open('360/json_pre_contest/train.json'))
# test = json.load(open('360/json_pre_contest/test.json'))
#
# cate = test['categories']
# result = []
#
# for ind, item in enumerate(cate):
#     result.append(translator.translate(item['name']))
# np.save('1_train_name.npy', np.array(result[:233]))
# np.save('1_test_name.npy', np.array(result))

############################################################# 人工修正翻译
# result = np.load('1_test_name.npy')
# result[16] = 'Scooter'  # Scooter|Scooters
# result[58] = 'all-in-one machine'  # 一体机
# result[74] = 'Soccer'  # Soccer/football
# result[115] = 'cutting board'  # a choopping block (= for cutting food on)
# result[157] = 'car'  # 汽车
# result[174] = 'Dragon fruit'  # 火龙果
# result[200] = 'Dumpling'  # Jiaozi
# result[201] = 'Bibcock'  # Lung Thau
#
# np.save('1_train_name.npy', np.array(result[:233]))
# np.save('1_test_name.npy', np.array(result))


############################################################# 进一步修正翻译
info = json.load(open('datasets/360/5_test_cat_info.json'))
result = [x['en_name'] for x in info]

np.save('1_test_name_new.npy', np.array(result))