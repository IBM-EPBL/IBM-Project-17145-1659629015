from flask import render_template, Flask, request
import numpy as np
import pickle
from sklearn.preprocessing import scale
app = Flask(__name__, template_folder='templates')

model = pickle.load(open("C:\\Users\\subba\\Downloads\\IBM-Project-10223-1659114439-main\\IBM-Project-10223-1659114439-main\\Final Deliverables\\implementation\\model.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login.html')
@app.route('/home.html')
def home1():
    return render_template('home.html')


@app.route('/prediction.html')
def formpg():
    return render_template('prediction.html')


@app.route('/prediction.html', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['Name']
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = request.form['credit']
        proparea = request.form['proparea']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    if married == 'Yes':
        married = 1
    else:
        married = 0

    if education == 'Graduate':
        education = 0
    else:
        education = 1

    if employed == 'Yes':
        employed = 1
    else:
        employed = 0

    if dependents == '3+':
        dependents = 3
    if credit == 'Yes':
        credit = 1
    else:
        credit = 0
    if proparea == 'Urban':
        proparea = 2
    elif proparea == 'Rural':
        proparea = 0
    else:
        proparea = 1

    features = [gender, married, dependents, education, employed, ApplicantIncome,
                CoapplicantIncome, LoanAmount, Loan_Amount_Term, credit, proparea]

    con_features = [np.array(features)]

    prediction = model.predict(con_features)
    print(prediction)
    if prediction == 1:
        return render_template('approve.html', prediction_text='Congratulations! '+name+' You are eligible for loan')
    else:
        return render_template('reject.html', prediction_text='Sorry '+name+' You are not eligible for loan')


if __name__ == "__main__":
    app.run(debug=True)
