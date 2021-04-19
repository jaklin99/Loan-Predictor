from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.embed import components

app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    #loan_id =request.form['Loan_ID']
    gender=request.form['Gender']
    married=request.form['Married']
    education=request.form['Education']
    self_eployed=request.form['Self_Employed']
    applicantIncome=request.form['ApllicantIncome']
    coapplicantIncome=request.form['CoapplicantIncome']
    loanAmountTerm=request.form['Loan_Amount_Term']
    credit_history=request.form['Credit_History']
    property_area=request.form['Propery_Area']
    loanAmount=request.form['Loan_Status']
    loanAmount_log=request.form['LoanAmount_log']
    totalIncome=request.form['TotalIncome']
    totalIncome_log=request.form['TotalIncome_log']
    #  creating a json object to hold the data from the form
    input_data=[{
    #'Loan_ID':loan_id,
    'Gender':gender,
    'Married' :married,
    'Education':education,
    'Self_Employed':self_eployed,
    'ApllicantIncome':applicantIncome,
    'CoapplicantIncome':coapplicantIncome,
    'Loan_Amount_Term':loanAmountTerm,
    'Credit_History':credit_history,
    'Propery_Area':property_area,
    'Loan_Status':loanAmount,
    'LoanAmount_log':loanAmount_log,
    'TotalIncome':totalIncome,
    'TotalIncome_log':totalIncome_log}]

    dataset=pd.DataFrame(input_data)

    dataset=dataset.rename(columns={
        #'Loan_ID': 'loan_id',
        'Gender': 'gender',
        'Married': 'married',
        'Education': 'education',
        'Self_Employed': 'self_eployed',
        'ApllicantIncome': 'applicantIncome',
        'CoapplicantIncome': 'coapplicantIncome',
        'Loan_Amount_Term': 'loanAmountTerm',
        'Credit_History': 'credit_history',
        'Propery_Area': 'property_area',
        'Loan_Status': 'loanAmount',
        'LoanAmount_log': 'loanAmount_log',
        'TotalIncome': 'totalIncome',
        'TotalIncome_log': 'totalIncome_log'})
    #'Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'ApllicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History', 'Propery_Area', 'Loan_Status', 'LoanAmount_log', 'TotalIncome', 'TotalIncome_log'

    dataset[['Gender', 'Married', 'Education', 'Self_Employed', 'ApllicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History',
             'Propery_Area', 'Loan_Status', 'LoanAmount_log', 'TotalIncome', 'TotalIncome_log']] = dataset[['Gender', 'Married', 'Education', 'Self_Employed', 'ApllicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History', 'Propery_Area', 'Loan_Status', 'LoanAmount_log', 'TotalIncome', 'TotalIncome_log']].astype(float)

    #dataset[['Term','Years in current job','Home Ownership','Purpose']]=dataset[['Term','Years in current job','Home Ownership','Purpose']].astype('object')

    dataset = dataset[['Gender', 'Married', 'Education', 'Self_Employed', 'ApllicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History', 'Propery_Area', 'Loan_Status', 'LoanAmount_log', 'TotalIncome', 'TotalIncome_log']]
    model = pickle.load(open('classifier.pkl', 'rb'))
    classifier=model.predict_proba(dataset)
    predictions = [item for sublist in classifier for item in sublist]
    colors = ['#1f77b4','#ff7f0e']
    loan_status = ['No','Yes']
    source = ColumnDataSource(
        data=dict(loan_status=loan_status, predictions=predictions))

    p = figure(x_range=loan_status, plot_height=500,
               toolbar_location=None, title="Loan Status", plot_width=800)
    p.vbar(x='loan_status', top='predictions', width=0.4, source=source, legend="loan_status",
           line_color='black', fill_color=factor_cmap('loan_status', palette=colors, factors=loan_status))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0.1
    p.y_range.end = 0.9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = 'Loan Status'
    p.yaxis.axis_label = ' Predicted Probabilities'
    script, div = components(p)
    return render_template('results.html',script=script,div=div)




if __name__=="__main__":
    app.run(debug=True)