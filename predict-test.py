import requests

url = 'http://localhost:9696/predict'

car = {
    "vehicle_class": "MID-SIZE",
    "engine_size(l)": 3.6,
    "transmission": "AS9",
    "fuel_type": "X",
    "fuel_consumption_comb_(l/100_km)": 9.9,
    "fuel_consumption_comb_(mpg)": 29
}

response = requests.post(url, json=car).json()
print(response)