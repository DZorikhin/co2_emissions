# Load the model

import pickle
import xgboost as xgb
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_eta=0.1_max_depth=5_min_ch_w=5.bin'

with open(model_file, 'rb') as f_in:
    dv, scaler, model = pickle.load(f_in)

app = Flask('emissions')

@app.route('/predict', methods=['POST'])
def predict():
    car = request.get_json()
    
    car['engine_size(l)'], car['fuel_consumption_comb_(l/100_km)'], car['fuel_consumption_comb_(mpg)'] = scaler.transform([[car['engine_size(l)'],
                                                                                                                            car['fuel_consumption_comb_(l/100_km)'], 
                                                                                                                            car['fuel_consumption_comb_(mpg)']]])[0]

    X = dv.transform([car])
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    
    xgb_pred = model.predict(dtest)
    
    result = {
        'CO2 emissions (g/km)': int(np.expm1(xgb_pred))
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)