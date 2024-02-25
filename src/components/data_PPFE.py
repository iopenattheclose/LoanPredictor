import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataPreProcessingFEConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataPreProcessingFE:
    def __init__(self):
        self.preprocessing_config=DataPreProcessingFEConfig()

    def initiate_data_preprocessing(self):
            logging.info("Entered the data preprocessing method or component")
            try:
                data=pd.read_csv('artifacts/data.csv')
                logging.info('Read the raw dataset needed for pre processing')
                
                #modifying columns as per EDA file
                data.drop(columns=['installment'], axis=1, inplace=True)
                data.loc[(data.home_ownership == 'ANY') | (data.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'
                data['issue_d'] = pd.to_datetime(data['issue_d'])
                data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
                data['title'] = data.title.str.lower()

                data['pub_rec'] = data['pub_rec'].apply(self.encode)
                data['mort_acc'] = data['mort_acc'].apply(self.encode)
                data['pub_rec_bankruptcies'] = data['pub_rec_bankruptcies'].apply(self.encode)
                data['loan_status'] = data['loan_status'].apply(self.encode)


                # Saving mean of mort_acc according to total_acc_avg 
                self.total_acc_avg = data.groupby(by='total_acc').mean(numeric_only=True).mort_acc
                data['mort_acc'] = data.apply(lambda x: self.fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
                data.dropna(inplace=True)

                numerical_data = data.select_dtypes(include='number')
                num_cols = numerical_data.columns

                for col in num_cols:
                    mean = data[col].mean()
                    std = data[col].std()

                    upper_limit = mean+3*std
                    lower_limit = mean-3*std

                    data = data[(data[col]<upper_limit) & (data[col]>lower_limit)]

                term_values = {' 36 months': 36, ' 60 months': 60}
                data['term'] = data.term.map(term_values)
                list_status = {'w': 0, 'f': 1}
                data['initial_list_status'] = data.initial_list_status.map(list_status)

                data.drop(columns=['issue_d', 'emp_title', 'title', 'sub_grade',
                   'address', 'earliest_cr_line', 'emp_length'],
                   axis=1, inplace=True)

                logging.info("Train test split initiated")

                train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)

                train_set.to_csv(self.preprocessing_config.train_data_path,index=False,header=True)

                test_set.to_csv(self.preprocessing_config.test_data_path,index=False,header=True)

                logging.info("Data split is completed")

                return(
                    data,
                    self.preprocessing_config.train_data_path,
                    self.preprocessing_config.test_data_path
                )
            
            except Exception as e:
                raise CustomException(e,sys)

    def encode(self,col_value):
        if col_value == 0.0 or col_value == "Fully Paid":
            return 0
        else:
            return 1
        
    def fill_mort_acc(self,total_acc, mort_acc):
        if np.isnan(mort_acc):
            return self.total_acc_avg[total_acc].round()
        else:
            return mort_acc
