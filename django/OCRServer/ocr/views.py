from django.shortcuts import render
from django.http import HttpResponse
import json
from PIL import Image
from io import BytesIO
import base64
import requests
import hashlib
import urllib.parse
import time

def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

def ocr(request):
    if request.method == 'POST':
        img = request.POST.get('img')
        if (img == ''):
            return HttpResponse('null')
        img = img.partition(',')[-1]

        url = "http://deepi.sogou.com:80/api/sogouService"
        pid = "7f37f594e2b22c13ee73c656ca6105dd"
        service = "basicOpenOcr"
        salt = str(time.time())
        SecretKey = "03ee22d82c65dc45435da716bce42994"
        # base64 string file picture,too long in the case we will omit string
        imageShort = img[0:1024]
        sign = md5(pid+service+salt+imageShort+SecretKey);
        payload = "lang=zh-CHS&pid=" + pid + "&service=" + service + "&sign=" + sign + "&salt=" + salt + "&image=" + urllib.parse.quote(img)
        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
            }
        response = requests.request("POST", url, data=payload, headers=headers)
        #print(response.text)
        return HttpResponse(response.text)