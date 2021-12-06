# About

The global transportation sector is a major polluter and in 2020 produced approximately 7.3 billion metric tons of carbon dioxide (CO2) emissions. Passenger cars were the biggest source of emissions that year, accounting for 41 percent of global transportation emissions.

In recent decades, global CO2 emissions from passenger cars have increased and peaked at 3.2 billion metric tons in 2019. Car emissions fell in 2020, but this was only due to the COVID-19 pandemic. Medium and heavy trucks are the second-largest polluters, accounting for 22 percent of transportation emissions. Although this was half the emissions of passenger cars, there are considerably fewer trucks on the road, showing just how polluting global road freight is. In 2020, heavy-duty truck CO2 emissions totaled almost two billion metric tons.

The idea of this project is to build the model in order to predict CO2 emissions of cars with certain technical characteristics.

# Dataset

Dataset provides model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

The dataset used in this project has been taken from [Kaggle](https://www.kaggle.com/debajyotipodder/co2-emission-by-vehicles?select=CO2+Emissions_Canada.csv). This Kaggle dataset is a compiled version of data provided by Government of Canada ([Original data](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6)) and contains data for 7 years. Also, dataset `co2_emissions.csv` and data description `data_description.csv` files have been uploaded to the project repository.

## Variables

<ol>
  <li>Make (string) - car producer</li>
  <li>Model (string) - car model</li>
  <li>Vehicle Class (string) - type of vehicle</li>
  <li>Engine Size(L) (float) - volume of engine</li>
  <li>Cylinders (integer) - number of cylinders</li>
  <li>Transmission (string) - type of transmission
    <ul>
      <li>A = Automatic</li>
      <li>AM = Automated manual</li>
      <li>AS = Automatic with select shift</li>
      <li>AV = Continuously variable</li>
      <li>M = Manual</li>
      <li>3 - 10 = Number of gears</li>
    </ul>
  </li>
  <li>Fuel Type (string)
    <ul>
      <li>X = Regular gasoline</li>
      <li>Z = Premium gasoline</li>
      <li>D = Diesel</li>
      <li>E = Ethanol (E85)</li>
      <li>N = Natural gas</li>
    </ul>
  </li>
  <li>Fuel Consumption City (L/100 km) (float)</li>
  <li>Fuel Consumption Hwy (L/100 km) (float)</li>
  <li>Fuel Consumption Comb (L/100 km) (float)</li>
  <li>Fuel Consumption Comb (mpg) (int)</li>
  <li>CO2 Emissions(g/km) (integer) - target variable</li>
</ol>

# Solution

This is a regression problem. It has been decided to use following models for the development of ML solution:

1. [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)</li>
2. [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
3. [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
4. [XGBoost Regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=regression#xgboost.XGBRFRegressor)
5. [Regression with Keras (TensorFlow)](https://www.tensorflow.org/api_docs)

Evaluation metric [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) has been chosen for the model selection process.

Please refer to `notebook.ipynb` where you may find the following:

- data preparation, data cleaning
- handling NaN values if any
- EDA, outliers analysis
- feature importance analysis, analysis of target variable
- models training and parameters tuning
- model selection process

Selected model is XGBoost Regressor based on chosen metric.

**Please be aware that this project has been done on macOS.**

# Train the final model and Predict (Flask deployment)

In `train.py` file you may find data preprocessing, training the final model and saving it into a pickle file. `predict.py` file contains loading the model and serving it via a web service with Flask. When you run Flask application with the model you could test it by running `python predict-test.py` and get the following prediction `{'CO2 emissions (g/km)': 232}`.

# Dependency and Environment management

I decided to use `pipenv` as virtual environment manager. In order to run my Flask application with model for prediction the following libraries are required:
- numpy
- flask
- gunicorn
- scikit-learn=0.24.2
- xgboost=1.4.2

The commands required to install `pipenv`, install all required dependencies and activate `pipenv`:
1. `pip install pipenv` - pipenv installation
2. `pipenv install numpy flask gunicorn scikit-learn==0.24.2 xgboost==1.4.2` - required dependencies installation
3. `pipenv shell` - activate the environment
4. `gunicorn --bind 0.0.0.0:9696 predict:app` - run the server with model to predict
5. `python predict-test.py` - run in separate terminal to test server is working and predict as expected

As the result two files were created: `Pipfile` and `Pipfile.lock`. These files are added to the current repository.

# Contanerization with Docker

I created `Dockerfile` which consists of the following:
```
FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_eta=0.1_max_depth=5_min_ch_w=5.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```

I created docker image by running the following command `docker build -t emissions .` and run docker container by executing following command `docker run -it --rm -p 9696:9696 emissions`. I tested the server running from the docker container and prediction by running this command from separate terminal `python predict-test.py`.

# Deployment to the Cloud: Heroku

*It is assumed that you already have Heroku account.*

Due to the nature of how Heroku creates applications and specific requirements to the Dockerfile command I created separate folder with heroku-specific Dockerfile. You could find it in this repository under `cloud-deployment/` folder. 

The content of the `Dockerfile` is different by only removing `EXPOSURE 9696` line and modifying last command which is specific to cloud deployment on Heroku `CMD gunicorn --bind 0.0.0.0:<dollar_sign>PORT predict:app`. The whole Dockerfile for cloud deployment:
```
FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_eta=0.1_max_depth=5_min_ch_w=5.bin", "./"]

CMD gunicorn --bind 0.0.0.0:<dollar_sign>PORT predict:app
```
## How to deploy app to Heroku

Make sure that you’re logged in to Heroku (`heroku login`) and then login to the Container Registry by running `heroku container:login`. Run the following commands:

- `heroku apps:create emissions-prediction-dz` - create app with any name
- `heroku container:push web` - build the image and push to Container Registry
- `heroku container:release web` - release the image to your app

## How to test app on Heroku

Public endpoint to test: `https://emissions-prediction-dz.herokuapp.com/predict`. Please run the command `python predict-test-cloud.py` to test the model prediction. Based on technical car data provided in that file you should get the following result:
```
{'CO2 emissions (g/km)': 337}
```
**Be aware that it might take some time (couple of minutes) for the app to respond because it might require to launch the app on Heroku platform.**