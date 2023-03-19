import sys
import os
import pandas as pd   
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.Exception import CustomException
from dataclasses import dataclass


from src.components.data_transformation import Data_transformation
from src.components.data_transformation import DataTransformation_config

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class dataingestion_config():
    train_path:str  = os.path.join("artifacts","train.csv")
    test_path:str = os.path.join("artifacts","test.csv")
    raw_path:str = os.path.join("artifacts","raw_data.csv")


class Data_ingestion:
    def __init__(self):
        self.ingestion_config = dataingestion_config()
        

    def initiate_data_ingestion(self):
        logging.info("Entering Data Ingestion")
        try:
             df = pd.read_csv("notebook/data/data.csv")
             logging.info("Reading the dataset from local file as CSV")
             os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
             df.to_csv(self.ingestion_config.raw_path,index=False,header=True)
             logging.info("Splittling Raw data Into train and Test data")
             train_set,test_set = train_test_split(df,test_size=0.2,random_state = 21)
             train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
             test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
             logging.info("Ingestion and Splitting of data is completed!!")
             return(
                  self.ingestion_config.train_path,
                  self.ingestion_config.test_path
             )
        except Exception as e:
                raise CustomException(e,sys)
                

if __name__ =="__main__":
     obj=Data_ingestion()
     train_data,test_data = obj.initiate_data_ingestion()
     data_transformation = Data_transformation()
     train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

     modeltrainer = ModelTrainer()
     print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



        

