import numpy as np
import pickle
from flask import Flask, render_template, request
model = pickle.load(open('rfc.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/inner') 
def inner():
  return render_template('inner-page.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    Administrative = request.form['Administrative']
    Administrative_Duration = request.form['Administrative_Duration']
    Informational = request.form['Informational']
    Informational_Duration = request.form['Informational_Duration']
    ProductRelated = request.form['ProductRelated']
    ProductRelated_Duration = request.form['ProductRelated_Duration']
    BounceRates = request.form['BounceRates']
    ExitRates = request.form['ExitRates']
    PageValues = request.form['PageValues']
    SpecialDay = request.form['SpecialDay']
    Month = request.form['Month']
    OperatingSystems = request.form['OperatingSystems']
    Browser = request.form['Browser']
    Region = request.form['Region']
    TrafficType = request.form['TrafficType']
    VisitorType = request.form['VisitorType']
    Weekend = request.form['Weekend']
    total = [[int(Administrative), float(Administrative_Duration), int(Informational), float(Informational_Duration),
              int(ProductRelated), float(ProductRelated_Duration), float(
                  BounceRates), float(ExitRates),
              float(PageValues), float(SpecialDay), int(
                  Month), int(OperatingSystems),
              int(Browser), int(Region), int(
                  TrafficType), int(VisitorType), int(Weekend)
              ]]
    total = scaler.inverse_transform(total)          
    print(total)
    prediction = model.predict_proba(total)
    print(prediction)
    # if prediction == 0:
    #     text = 'The visitor is not interested in buying products'
    # else:
    #     text = 'The visitor is intereted in buying products'
    text = f"The probability of the visitor buying is {prediction[0][0]}"

    return render_template('inner-page.html', prediction_text=text)

    
if __name__ == '__main__':
    app.run(debug=True)