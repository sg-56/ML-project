import os
import sys
import numpy as np 
import pandas as pd 
import dill
from src.Exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        pass
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv = 3,n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train) ## Training Model

            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            train_model_score = r2_score(y_train,train_preds)
            test_model_score = r2_score(y_test,test_preds)

            report[list(models.keys())[i]] = test_model_score

        return report
        pass
    except Exception as e:
        raise CustomException(e,sys)
        


def load_object(file_path:str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
