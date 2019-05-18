from flask import Flask, request
from flask_restful import Api
from evaluator.classifier_evaluator import Evaluator
import json

app = Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == "POST":
        data = request.json
        evl = Evaluator()
        predicted = evl.test(data['data'])
        return predicted.to_json(orient='records', force_ascii=False)


@app.route('/train', methods=['GET', 'POST'])
def train():
        evl = Evaluator()
        accuracy = evl.train()
        return json.dumps(accuracy)


if __name__ == '__main__':
    app.run()
