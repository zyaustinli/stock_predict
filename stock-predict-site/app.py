from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import matplotlib.pyplot as plt
from stock_prediction import data_pre 

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    stock_name = data['stockName']
    forecast_out = int(data['forecastDays'])
    model_num = int(data['modelType'])

    fig = data_pre(stock_name, forecast_out, model_num)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)