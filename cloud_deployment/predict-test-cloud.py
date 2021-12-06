import requests

host = 'https://emissions-prediction-dz.herokuapp.com/'
url = f'{host}/predict'

car = {
    "vehicle_class": "SUV - SMALL",
    "engine_size(l)": 3,
    "transmission": "AS8",
    "fuel_type": "Z",
    "fuel_consumption_comb_(l/100_km)": 14.6,
    "fuel_consumption_comb_(mpg)": 19
}

response = requests.post(url, json=car).json()
print(response)