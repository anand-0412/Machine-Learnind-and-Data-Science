from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

std_scaler = pickle.load(open("scaler.pkl","rb"))
tree_model = pickle.load(open("model_1.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predictdata",methods=["GET","POST"])
def pred_datapoint():
    if request.method=="POST":
        Age	= int(request.form.get("Age"))
        Sex	= int(request.form.get("Sex"))
        Chest_pain_type	= int(request.form.get("Chest pain type"))
        BP	= int(request.form.get("BP"))
        Cholesterol	= int(request.form.get("Cholesterol"))
        FBS_over_120	= int(request.form.get("FBS over 120"))
        EKG_results	= int(request.form.get("EKG results"))
        Exercise_angina	= int(request.form.get("Exercise angina"))
        ST_depression	= float(request.form.get("ST depression"))
        Slope_of_ST	= int(request.form.get("Slope of ST"))
        Number_of_vessels_fluro	= int(request.form.get("Number of vessels fluro"))
        Thallium =int(request.form.get("Thallium"))

        new_data_scaled = std_scaler.transform([[Age,Sex,Chest_pain_type,BP,Cholesterol,FBS_over_120,
                                                EKG_results,Exercise_angina,ST_depression,Slope_of_ST, 
                                                Number_of_vessels_fluro, Thallium]])
        result = tree_model.predict(new_data_scaled)

        return render_template("home.html",result=result)
    else: 
        return render_template("home.html")
if __name__=="__main__":
    app.run(debug=True)


'''
Age	Sex	Chest pain type	BP	Cholesterol	FBS over 120	EKG results	Exercise angina	ST depression	Slope of ST	Number of vessels fluro	Thallium
'''