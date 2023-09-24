import os.path
import urllib.request
import re
import json
import time

test_info = json.load(open('datasets/360/test.json'))
# test_name2id = {x['name']: x['id'] for x in test_info['categories'][-233:]}
test_name2id = {x['name']: x['id'] for x in test_info['categories'][-41:]}

headers = ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0")
opener = urllib.request.build_opener()
opener.addheaders = [headers]
urllib.request.install_opener(opener)

page_num = 1
for nam, idx in test_name2id.items():
    keyname = nam
    key = urllib.request.quote(keyname)
    for i in range(1, page_num+1):
        url = "https://search.jd.com/Search?keyword=" + key + "&wq=" + key + "&page=" + str(i * 2 - 1);
        data = urllib.request.urlopen(url).read().decode("utf-8", "ignore")
        pat = 'data-lazy-img="(.*?)"'
        imagelist = re.compile(pat).findall(data)
        print(f'共有{len(imagelist)}张图片')
        for j in range(0, len(imagelist)):
            b1 = imagelist[j].replace('/n7', '/n0')
            print("第" + str(i) + "页第" + str(j) + "张爬取成功")
            newurl = "http:" + b1
            file_dir = f"data/novel_image/{idx}/"
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file = file_dir + str(j) + '.jpg'
            try:
                urllib.request.urlretrieve(newurl, filename=file)
            except:
                pass
            time.sleep(0.1)
        time.sleep(1)
    time.sleep(2)
