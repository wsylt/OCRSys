from PIL import Image
from flask_cors import CORS
from flask import Flask, request
from io import BytesIO
import base64
import requests
import hashlib
import urllib.parse 
import json
import time
import segmentation

def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

app = Flask(__name__)
CORS(app, supports_credentials=True)
 
@app.route('/ocr', methods=['GET','POST'])
def ocr():
    img = request.json.get('img')
    if (img == ''):
        return 'null'
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
    
    return (response.text)

@app.route('/segment', methods=['GET','POST'])
def seg():
    return segmentation.segment(request.json.get('data'), './dic.json')

    # dic = json.loads(response.text)
    # txt = ''
    # for i in dic['result']:
    # 	txt += i['content']
    # #return txt
    # #print (json.dumps(dic, indent=2))
    # #return json.dumps(dic, indent=2)

if __name__ == '__main__':
    app.run(port=5000)
