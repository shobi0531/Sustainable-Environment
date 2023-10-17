from flask import Flask, render_template, request
import pandas as pd
 # Import the model class

app = Flask(__name__,template_folder='template')


import numpy as np
import seaborn as sns


df = pd.read_csv('air pollution.csv',na_values='=')


data2=df.copy()
data2=data2.fillna(data2.mean())
#mapping
dist=(data2['City'])
distset=set(dist)
dd=list(distset)
dict0fWords= {dd[i] : i for i in range(0, len(dd))}
data2['City']=data2['City'].map(dict0fWords)
dist=(data2['AQI_Bucket'])
distset=set(dist)
dd=list(distset)
dict0fWords= {dd[i] : i for i in range(0, len(dd))}
data2['AQI_Bucket']=data2['AQI_Bucket'].map(dict0fWords)
data2['AQI_Bucket']=data2['AQI_Bucket'].fillna(data2['AQI_Bucket'].mean())
features=data2[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3','Benzene', 'Toluene', 'Xylene']]
labels=data2['AQI']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr=RandomForestRegressor(max_depth=2,random_state=0)
regr.fit(Xtrain,Ytrain)
y_pred=regr.predict(Xtest)
from sklearn.metrics import r2_score
r2_score(Ytest,y_pred)

# Function to get AQI rating
def get_aqi_rating(aqi):
    if 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Satisfactory"
    elif 101 <= aqi <= 200:
        return "Moderate"
    elif 201 <= aqi <= 300:
        return "Poor"
    elif 301 <= aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        city = request.form['location']
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no = float(request.form['no'])
        no2 = float(request.form['no2'])
        nox = float(request.form['nox'])
        nh3 = float(request.form['nh3'])
        co = float(request.form['co'])
        so2 = float(request.form['so2'])
        o3 = float(request.form['o3'])
        benzene = float(request.form['benzene'])
        toluene = float(request.form['toluene'])
        xylene = float(request.form['xylene'])
        
        # Create a DataFrame with user input
        user_data = pd.DataFrame({
            'City': [city],
            'PM2.5': [pm25],
            'PM10': [pm10],
            'NO': [no],
            'NO2': [no2],
            'NOx': [nox],
            'NH3': [nh3],
            'CO': [co],
            'SO2': [so2],
            'O3': [o3],
            'Benzene': [benzene],
            'Toluene': [toluene],
            'Xylene': [xylene]
            # Add more columns for other gases as needed
        })

        # Map the city value to match the mapping used during training
        dict0fWords = {'Ahmedabad': 0, 'Delhi': 1, 'Mumbai': 2, 'Chennai': 3}  # Update with your city mapping
        user_data['City'] = user_data['City'].map(dict0fWords)

        # Use the loaded model to make predictions
        predicted_aqi = regr.predict(user_data)[0]
        predicted_rating = get_aqi_rating(predicted_aqi)

        return render_template('air.html', aqi=predicted_aqi, rating=predicted_rating)

    return render_template('air.html')

if __name__ == '__main__':
    app.run(debug=True)
