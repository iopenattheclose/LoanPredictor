import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(  self,
        loan_amnt: int,
        term: int,
        int_rate:int,
        grade: str,
        home_ownership: str,
        annual_inc: int,
        verification_status: str,
        purpose: str,
        dti:int,
        open_acc:int,
        pub_rec:int,
        revol_bal:int,
        revol_util:int,
        total_acc:int,
        initial_list_status:int,
        application_type:str,
        mort_acc:int,
        pub_rec_bankruptcies:int
        ):
        self.term = term
        self.loan_amnt = loan_amnt
        self.int_rate = int_rate
        self.grade = grade
        self.home_ownership = home_ownership
        self.annual_inc = annual_inc
        self.verification_status = verification_status
        self.purpose = purpose
        self.dti = dti
        self.open_acc = open_acc
        self.pub_rec = pub_rec
        self.revol_bal = revol_bal
        self.revol_util = revol_util
        self.total_acc = total_acc
        self.initial_list_status = initial_list_status
        self.application_type = application_type
        self.mort_acc = mort_acc
        self.pub_rec_bankruptcies = pub_rec_bankruptcies

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
               "loan_amnt": [self.loan_amnt],
        "term": [self.term],
        "int_rate":[self.int_rate],
        "grade": [self.grade],
        "home_ownership": [self.home_ownership],
        "annual_inc": [self.annual_inc],
        "verification_status": [self.verification_status],
        "purpose": [self.purpose],
        "dti":[self.dti],
        "open_acc":[self.open_acc],
        "pub_rec":[self.pub_rec],
        "revol_bal":[self.revol_bal],
        "revol_util":[self.revol_util],
        "total_acc":[self.total_acc],
        "initial_list_status":[self.initial_list_status],
        "application_type":[self.application_type],
        "mort_acc":[self.mort_acc],
        "pub_rec_bankruptcies":[self.pub_rec_bankruptcies]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)