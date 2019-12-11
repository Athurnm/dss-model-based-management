from flask import Flask, jsonify, request
import pickle
import numpy as np
import pandas as pd
import datetime
from fbprophet import Prophet

# test num
cols = ['Mie Ayam','Bakso Goreng','Bakso Keju']

# server
app = Flask(__name__)

# routes
@app.route('/', methods=['GET'])
def index():
    return '''<h1>Food Demand Prediction API/h1>
<p>Working!!!</p>'''

@app.route('/predict', methods=['POST'])
def predict():
    # get date
    date = request.get_json(force=True)['date']
    target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    today = (target_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    twodays = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Create and return prediction
    results = []
    for col in cols:
        model = pickle.load(open('model'+'_'+col+'.pickle', 'rb'))
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

    tf_revised = tf_revised[['Mie Ayam','Bakso Goreng','Bakso Keju']]
    exp = tf_revised.to_dict(orient="records")
    
    output = []
    for col in cols:
        les = {}
        les['itemName'] = col
        les['quantity'] = exp[0][col]
        output.append(les)
    
    return jsonify(output),200

if __name__ == '__main__':
    app.run(port = 5000, debug=True)