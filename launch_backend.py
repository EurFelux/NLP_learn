import torch
from flask import Flask, request, render_template
from mymodel.classes import *
from mymodel.utils import generate_answer, tokenizer, device

app = Flask('ChatBot')
model_path = 'models/chatbot_lr=0.0004_DecLayer=5.pth'

try:
    model = torch.load(model_path)
except:
    exit()


@app.route('/question', methods=['POST'])
def chatbot():
    text = request.data.decode(encoding='utf8')
    res = generate_answer(text, tokenizer, model, device)
    headers = {'Access-Control-Allow-Origin': 'http://127.0.0.1:5500'}
    return ({'data': res}, headers)


@app.route('/', methods=['GET', 'POST'])
def front_end():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
