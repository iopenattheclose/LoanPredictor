import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MaxAbsScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            logreg = LogisticRegression(max_iter=1000)
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logreg.fit(X_train, y_train)

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = logreg
            )

            y_pred = logreg.predict(X_test)
            print("Max value of ypred array is :",y_pred.max())
            print(np.unique(y_pred, return_counts=True))

            # X = MaxAbsScaler().fit_transform(X)
            # kfold =     KFold(n_splits=5)
            # accuracy = np.mean(cross_val_score(logreg, X, y, cv=kfold, scoring='accuracy', n_jobs=-1))
            # print("Cross Validation accuracy: {:.3f}".format(accuracy))

            final_score = logreg.score(X_test, y_test)
            return final_score

        except Exception as e:
            raise CustomException(e,sys)