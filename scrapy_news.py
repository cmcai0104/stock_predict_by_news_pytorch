import time
from datetime import datetime
import json
import hashlib
from urllib.request import urlopen, Request


def USE_SHA(text):
    if not isinstance(text, bytes):
        text = bytes(text, 'utf-8')
    sha = hashlib.sha1(text)
    encrypts = sha.hexdigest()
    return encrypts


def md5value(s):
    return hashlib.md5(s.encode()).hexdigest()


def scrapy(stamp):
    url = 'https://www.cls.cn/nodeapi/updateTelegraphList?app=CailianpressWeb&category=&hasFirstVipArticle=0&lastTime=%s&os=web&rn=20&subscribedColumnIds=&sv=7.5.5&' % stamp
    # url = 'https://www.cls.cn/v1/roll/get_roll_list?app=CailianpressWeb&category=&lastTime=%s&last_time=%s&os=web&refresh_type=1&rn=20&sv=7.5.5&' % stamp
    url = url + 'sign=' + md5value(USE_SHA(url))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
    ret = Request(url, headers=headers)
    response = urlopen(ret).read()
    html = response.decode('utf-8')
    json_data = json.loads(html)['data'] #json_data['roll_data']:list, len = json_data['update_num']
    content = json_data['roll_data'][0]['content']
    return content

if __name__ == '__main__':
    start_str = '2020-01-01 00:00:00'
    start_time = time.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    start_Stamp = int(time.mktime(start_time))

    with open('./data/rawdata2.txt', 'w', encoding='utf-8') as wf:
        content = ''
        while start_Stamp <= int(time.time()):
            text = scrapy(start_Stamp)
            if content !=  text:
                wf.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_Stamp)) + '##TAP##' +  text + ' \n')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_Stamp)) + '##TAP##' +  text + ' \n')
                content =  text
            start_Stamp += 60
