import os
import sys
import pandas as pd 
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
                                RandomForestRegressor,
                                AdaBoostRegressor,
                                GradientBoostingRegressor,
                            )

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging
from src.Exception import CustomException
from src.Utils import save_object
from src.Utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting Training and Test Input Data")
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],
                                            #train_array[:,-1],
                                            train_arr[:,-1],
                                            test_arr[:,:-1],
                                            test_arr[:,-1]                                             
                                             )
            models ={"Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    #"XGBRegressor": XGBRegressor(),
                    #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor()
                    }
            
            model_report:dict = evaluate_models(X_train=X_train,
                                               X_test = X_test,
                                               y_train=y_train,
                                               y_test = y_test,
                                               models = models,
                                               #params=params
                                               )
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
                                                                                                    
            if best_model_score < 0.6:
                raise CustomException("No best model Found")
            
            logging.info("Best Model Found on both train and testing Data")

            save_object(file_path=ModelTrainerConfig.trained_model_file_path,
                        obj=best_model) 
            predicted = best_model.predict(X_test)
            r_square = r2_score(y_test,predicted)
            return r_square
            logging.info("Training and DUmping of best Model Object file Completed")
            pass
        except Exception as e:
            raise CustomException(e,sys)
            


