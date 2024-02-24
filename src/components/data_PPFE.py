import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataPreProcessingConfig:
    #declaring three path variables which are created in artifacts folder
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataPreProcessing:
    def __init__(self):
        # ingestion_config variable will consist of the three values defined in DataIngestionConfig class
        self.preprocessing_config=DataPreProcessingConfig()

    def initiate_data_preprocessing(self):
            logging.info("Entered the data preprocessing method or component")
            try:
                #instead of csv file, this data can be fetched from any data source (viz mongodb)
                data=pd.read_csv('artifacts/data.csv')
                logging.info('Read the raw dataset needed for pre processing')
                
                #modifying columns as per EDA file
                data.drop(columns=['installment'], axis=1, inplace=True)
                data.loc[(data.home_ownership == 'ANY') | (data.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'
                data['issue_d'] = pd.to_datetime(data['issue_d'])
                data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
                print(data.head())

                print(data["loan_status"].head())

                data['pub_rec'] = data['pub_rec'].apply(self.encode)
                data['mort_acc'] = data['mort_acc'].apply(self.encode)
                data['pub_rec_bankruptcies'] = data['pub_rec_bankruptcies'].apply(self.encode)
                data['loan_status'] = data['loan_status'].apply(self.encode)

                print(data.head())
                print(data["loan_status"].head())

                # logging.info("Train test split initiated")
                # train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

                # train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

                # test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                # return(
                #      #returning path of train and test because this will be used in data transformationl
                #     self.ingestion_config.train_data_path,
                #     self.ingestion_config.test_data_path
                # )
            except Exception as e:
                raise CustomException(e,sys)

    def encode(self,col_value):
        if col_value == 0.0 or col_value == "Fully Paid":
            return 0
        else:
            return 1
        
if __name__=="__main__":
    obj=DataPreProcessing()
    obj.initiate_data_preprocessing()