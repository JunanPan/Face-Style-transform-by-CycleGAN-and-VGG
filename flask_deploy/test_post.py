from aiohttp import JsonPayload
from flask import Flask, render_template, request, make_response,jsonify
import json
from model_predict import predict
from flask_cors import CORS
from flask_cors import cross_origin
app = Flask(__name__)
CORS(app, resources=r'/*')
app.debug = True
@app.route('/', methods=['POST', 'GET'])
def main():
    return 'hello!'
@app.route('/test/', methods=['POST', 'GET'])
def add_func():
    return jsonify({'a':'hello!123','b':'456789'})
@app.route('/app/login/', methods=['POST'])
def predict_func():
    # print(request.get_data(as_text=True))
    front_reponse = request.get_data(as_text=True).strip()
    base64_code_str  =front_reponse#从前端拿到str类型的base64
    # base64_code_str  =front_reponse[22:]#从前端拿到str类型的base64
    base64_code_byte  =bytes(base64_code_str.encode())[22:]#转换成标准的 字节类型的base64
    #predict送入字节型base64，送出三个字符串型的base64
    # result_base64_str1,result_base64_str2,result_base64_str3 = predict(base64_code_byte)
    with open('./前端编码成base641.txt','wb') as f:
        f.write(base64_code_byte)
    result_base64_str1,result_base64_str2,result_base64_str3 = predict(base64_code_byte)
    return jsonify({
        "flag":1,
        "patingStyle":result_base64_str1,
        "comicStyle" :result_base64_str2,
        "sumiaoStyle":result_base64_str3
    })
if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0')
