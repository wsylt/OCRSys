from PIL import Image
import pytesseract
from flask_cors import CORS
from flask import Flask, request
from io import BytesIO
import base64
 
app = Flask(__name__)
CORS(app, supports_credentials=True)
 
@app.route('/', methods=['GET','POST'])
def ocr():
    img = request.json.get('img')
    if (img == ''):
        return 'null'
    img = img.partition(',')[-1]
    #src = request.data['img']
    text = pytesseract.image_to_string(Image.open(BytesIO(base64.b64decode(img))), lang = 'chi_sim')
    return text
 
if __name__ == '__main__':
    app.run(port=5000)
