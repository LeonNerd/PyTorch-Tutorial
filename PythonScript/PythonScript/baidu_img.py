# coding=utf-8
"""根据搜索词下载百度图片"""
import re
import os
import time
import requests
from urllib.parse import quote
from multiprocessing import Pool
from xpinyin import Pinyin


def get_page(keyword, page, n):
    page = page * n
    keyword = quote(keyword, safe='/')
    url_begin = "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word="
    url = url_begin + keyword + "&pn=" + str(page) + "&gsm=" + str(hex(page)) + "&ct=&ic=0&lm=-1&width=0&height=0"
    print(url)
    return url


def get_onepage_urls(onepageurl):
    try:
        html = requests.get(onepageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        return pic_urls
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    return pic_urls


def down_pic(pic_urls, keyword):
    """给出图片链接列表, 下载所有图片"""
    print([keyword], ' ---total url num---', len(pic_urls), '----------')
    pin = Pinyin()
    try:
        keyword_pin = "".join(pin.get_pinyin(keyword.decode("utf-8")).split("-")).strip()
    except:
        keyword_pin = "".join(pin.get_pinyin(keyword).split("-")).strip()
    keyword_pin = '_'.join(keyword_pin.split())
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=20)
            string = str(i + 1) + '.jpg'
            if not os.path.exists(keyword_pin):
                os.makedirs(keyword_pin)
            with open(keyword_pin+'/'+keyword_pin+string, 'wb') as f:
                f.write(pic.content)
                print([keyword], '----- 成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print([keyword], '----- 下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(['ERROR'], e)
            continue


def run(keyword):
    page_begin = 0
    page_number = 20
    image_number = 100
    all_pic_urls = []
    while 1:
        if page_begin > image_number:
            break
        print([keyword], "----- 第%d次请求数据" % page_begin, [page_begin])
        url = get_page(keyword, page_begin, page_number)
        time.sleep(0.1)
        onepage_urls = get_onepage_urls(url)
        page_begin += 1

        all_pic_urls.extend(onepage_urls)

    down_pic(list(set(all_pic_urls)), keyword)


if __name__ == '__main__':
    # keywords = ['吊车 工地', '吊车 建筑', '吊车 电网', '吊车 施工', '吊车 电塔', '吊车 电线', '水泥泵车 工地', '水泥泵车 建筑',
    #             '水泥泵车 电网', '水泥泵车 施工', '水泥泵车 电塔', '水泥泵车 电线', '塔吊 工地', '塔吊 建筑', '塔吊 电网',
    #             '塔吊 施工', '塔吊 电塔', '塔吊 电线', '挖掘机 工地', '挖掘机 建筑', '挖掘机 电网', '挖掘机 施工',
    #             '挖掘机 电塔', '挖掘机 电线']
    keywords = [
        '猫', '狗', '兔子',
    ]

    p = Pool(processes=4)
    p.map(run, keywords)
    p.close()
    p.join()
