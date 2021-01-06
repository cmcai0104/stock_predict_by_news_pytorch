from loguru import logger
import hashlib
import time
from datetime import datetime
import json
from urllib.request import urlopen, Request


def text_sha(text):
    if not isinstance(text, bytes):
        text = bytes(text, 'utf-8')
    sha = hashlib.sha1(text)
    encrypts = sha.hexdigest()
    return encrypts

def md5(word):
    m = hashlib.md5()
    b = word.encode(encoding='utf-8')
    m.update(b)
    return m.hexdigest()


def generate_url(stamp):
    url = 'https://www.cls.cn/v1/roll/get_roll_list?'
    text = 'app=CailianpressWeb&category=&lastTime='+str(stamp)+'&last_time='+str(stamp)+'&os=web&refresh_type=1&rn=20&sv=7.5.5'
    sha_text = text_sha(text)
    md5_text = md5(sha_text)
    url = url + text + '&sign=' + md5_text
    return url


def scrapy(stamp):
    url = generate_url(stamp)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
    ret = Request(url, headers=headers)
    response = urlopen(ret).read()
    html = response.decode('utf-8')
    json_data = json.loads(html)['data'] #json_data['roll_data']:list, len = json_data['update_num']
    content = json_data['roll_data']
    return content


if __name__ == '__main__':
    start_time = time.strptime('2021-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    start_Stamp = int(time.mktime(start_time))
    min_stamp = start_Stamp

    end_time = time.strptime('2014-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    end_Stamp = int(time.mktime(end_time))

    with open('./data/history_news.txt', 'w', encoding='utf-8') as wf:
        wf.write('create_time#split#modified_time#split#title#split#reading_num#split#share_num#split#recommend_num#split#news_content\n')
        while start_Stamp >= end_Stamp:
            texts = scrapy(start_Stamp)
            for text in texts:
                news_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(text['ctime']))
                min_stamp = min(int(text['ctime']), min_stamp)
                news_modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(text['modified_time']))
                news_title = text['title'].replace('\n', '').replace('\r', '').strip()
                news_content = text['content'].replace('\n', '').replace('\r', '').strip()
                reading_num = text['reading_num']
                share_num = text['share_num']
                recommend_num = text['recommend']
                wf.write('%s#split#%s#split#%s#split#%s#split#%s#split#%s#split#%s\n'%(news_time, news_modified_time, news_title, reading_num, share_num, recommend_num, news_content))
                print('%s创建%s修改，%s，阅读%s，分享%s，推荐%s'%(news_time, news_modified_time, news_title, reading_num, share_num, recommend_num))
            start_Stamp = min_stamp
