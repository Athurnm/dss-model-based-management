# Data Handling
from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd
import datetime
from pydantic import BaseModel
from fbprophet import Prophet
import requests
from starlette.responses import Response

# PATH
DATAPATH = '../notebooks/'

# Server
app = FastAPI()

# Read dataset
tf = pd.read_csv('../notebooks/dss_mock_1.csv')
tf['date_conv'] = tf['date'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
tf.drop(['date', 'is_holiday'], axis=1, inplace=True)
tf['date_conv'] = pd.DatetimeIndex(tf['date_conv'])
tf = tf.sort_values(by='date_conv').reset_index(drop=True)

# list all column on dataset
cols = list(tf.columns.values)
cols.remove('date_conv')


class Date(BaseModel):
    date : str = None

def convertToBahan(tf_revised, cols):
    resp = requests.get("http://localhost:3000/API/item")
    bahan = resp.json()
    menu = [{}]
    for col in cols:
        ds = {}
        menu.append(ds)
    i = 0
    for col in cols:
        menu[i]['foodName'] = col
        menu[i]['ingredients'] = []
        for x in bahan:
            if x['itemName'] == col:
                les = []
                for a in x['ingredients']:
                    les = {'ingredientName':a['ingredientName'],'quantity':a['quantity']*tf_revised[col].values[0], 'metric':a['metric']}
                    menu[i]['ingredients'].append(les)
        i = i+1
    menu.pop()
    return menu


@app.post("/predict")
async def predict(date: Date):
    # getting current date and two days (should date not provided)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    twodays = (datetime.datetime.today() + datetime.timedelta(days=2)).strftime("%Y-%m-%d")

    if date.date is not None:
        target_date = datetime.datetime.strptime(date.date, "%Y-%m-%d")
        today = (target_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        twodays = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        print(today)

    # Create and return prediction
    results = []
    for col in cols:
        model = pickle.load(open(DATAPATH+'model'+'_'+col+'.pickle', 'rb'))
        f_df = model.make_future_dataframe(periods=30, freq='D')
        result = model.predict(f_df)
        result_tf = result[['ds', 'yhat']].rename(columns={'yhat': col})
        results.append(result_tf)
    prediction = pd.concat(results, axis=1)

    # Cleaning predict table
    prediction = prediction.iloc[:,  ~prediction.columns.duplicated()]
    prediction.drop_duplicates(subset='ds', keep="first", inplace=True)
    prediction = prediction.reset_index(
        drop=True).rename(columns={'ds': 'date'})
    
    for col in cols:
        prediction[col] = prediction[col].apply(np.floor).astype(int)

    tf_revised = prediction[(prediction['date'] > today) & (prediction['date'] < twodays)
                        ].reset_index(drop=True)
    # Convert to bahan
    menu_bahan = convertToBahan(tf_revised, cols)

    tf_revised = tf_revised[['Mie Ayam','Bakso Goreng','Bakso Keju']]
    exp = tf_revised.to_dict(orient="records")
    output = []
    for col in cols:
        les = {}
        les['itemName'] = col
        les['quantity'] = exp[0][col]
        output.append(les)

    return output

