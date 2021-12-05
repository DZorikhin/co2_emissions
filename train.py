import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


# data preparation

df = pd.read_csv('co2_emissions.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
y_full_train = df_full_train['co2_emissions(g/km)'].values
y_test = df_test['co2_emissions(g/km)'].values

selected_f = [
    'vehicle_class', 'engine_size(l)', 'transmission', 'fuel_type', 
    'fuel_consumption_comb_(l/100_km)', 'fuel_consumption_comb_(mpg)'
]

df_full_train = df_full_train[selected_f].reset_index(drop=True)
df_test = df_test[selected_f].reset_index(drop=True)

scaler = MinMaxScaler()

df_full_train[['engine_size(l)', 
               'fuel_consumption_comb_(l/100_km)', 
               'fuel_consumption_comb_(mpg)']] = scaler.fit_transform(df_full_train[['engine_size(l)', 
                                                                                     'fuel_consumption_comb_(l/100_km)', 
                                                                                     'fuel_consumption_comb_(mpg)']])
df_test[['engine_size(l)', 
          'fuel_consumption_comb_(l/100_km)', 
          'fuel_consumption_comb_(mpg)']] = scaler.fit_transform(df_test[['engine_size(l)', 
                                                                           'fuel_consumption_comb_(l/100_km)', 
                                                                           'fuel_consumption_comb_(mpg)']])
y_full_train = np.log1p(y_full_train)
y_test = np.log1p(y_test)


# parameters

eta = 0.1
max_depth = 5
min_child_weight = 5
output_file = f'model_eta={eta}_max_depth={max_depth}_min_ch_w={min_child_weight}.bin'


# training and prediction functions

def train(df_train, y_train, eta, max_depth, min_child_weight):
    
    dv = DictVectorizer(sparse=False)
    full_train_dicts = df_train.to_dict(orient='records')
    X_full_train = dv.fit_transform(full_train_dicts)
    dfulltrain = xgb.DMatrix(X_full_train, label=y_train, feature_names=dv.get_feature_names())
    
    xgb_params = {
        'eta': eta,
        'max_depth': max_depth, 
        'min_child_weight': min_child_weight,
    
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        
        'nthread': 8,
        'seed': 13,
        'verbosity': 1
    }
    
    model = xgb.train(xgb_params, dfulltrain, num_boost_round=44)
    
    return dv, model


def predict(df_test, y_test, dv, model):
    test_dict = df_test.to_dict(orient='records')

    X_test = dv.transform(test_dict)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dv.get_feature_names())
    
    xgb_pred = model.predict(dtest)

    return xgb_pred


# train the final model

print('training the final model')

dv, model = train(df_full_train, y_full_train, eta, max_depth, min_child_weight)
y_pred = predict(df_test, y_test, dv, model)

xgb_rmse = round(mean_squared_error(y_test, y_pred, squared=False), 4)
print(f'RMSE metric for XGBoost Regressor final model: {xgb_rmse}')


# save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')