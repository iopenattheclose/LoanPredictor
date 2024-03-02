from flask import Flask,request,render_template
import numpy as np
import pandas as pd



from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            loan_amnt=request.form.get('loan_amnt'),
            term=request.form.get('term'),
            int_rate=request.form.get('int_rate'),
            grade=request.form.get('grade'),
            home_ownership=request.form.get('home_ownership'),
            annual_inc=request.form.get('annual_inc'),
            verification_status=request.form.get('verification_status'),
            purpose=request.form.get('purpose'),
            dti=request.form.get('dti'),
            open_acc=request.form.get('open_acc'),
            pub_rec=request.form.get('pub_rec'),
            revol_bal=request.form.get('revol_bal'),
            revol_util=request.form.get('revol_util'),
            total_acc=request.form.get('total_acc'),
            initial_list_status=request.form.get('initial_list_status'),
            application_type=request.form.get('application_type'),
            mort_acc=request.form.get('mort_acc'),
            pub_rec_bankruptcies=request.form.get('pub_rec_bankruptcies')   
   )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        if results == 1:
            results="Sorry your loan will not be dispersed."
        else:
            results = "Congratulations, you will be granted a loan!!"
        return render_template('home.html',results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)        
    # app.run(host='0.0.0.0',port=5000, debug=True)#port=8080)
