import sys
import os
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.Exception import CustomException
from src.logger import logging
from src.Utils import save_object



@dataclass
class DataTransformation_config:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")


class Data_transformation:
    def __init__ (self):
        self.data_transformation_config = DataTransformation_config()

    def get_data_transormation_obj(self):
        try:
            numerical_cols = ['reading_score','writing_score']
            categorical_cols = ["gender",
                                "race_ethnicity",
                                "parental_level_of_education",
                                "lunch",
                                "test_preparation_course"
                                ]
            num_pipeline = Pipeline(steps = [
                                        ('Imputer',SimpleImputer(strategy='median')),
                                        ( 'scaler',StandardScaler())                                     
                                        ]
                                    )
            cat_pipeline = Pipeline(steps = [
                                                ('Imputer',SimpleImputer(strategy='most_frequent')),    
                                                ('OneHotEncoding',OneHotEncoder())
                                            ]
                                    )
            logging.info(f'Categorical_columns :{categorical_cols}')
            logging.info(f'Numerical COlumns : {numerical_cols}')

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipelines",cat_pipeline,categorical_cols)

                ]


            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_data = pd.read_csv(train_path)
                test_data = pd.read_csv(test_path)
                logging.info("Reading Train and Test Data Set completed")
                logging.info("Reading Preprocessing Object")
                preprocessor_obj = self.get_data_transormation_obj()
                target_column = ['math_score']
                numerical_cols = ['reading_score','writing_score']
                categorical_cols = ["gender",
                                "race_ethnicity",
                                "parental_level_of_education",
                                "lunch",
                                "test_preparation_course"
                                ]
                input_features_train_df = train_data.drop(target_column,axis=1)
                train_target_feature = train_data[target_column]


                input_features_test_df = test_data.drop(target_column,axis=1)
                test_target_feature = test_data[target_column]

                logging.info(f'Applying Preprocessing to train and test Data')

                input_feature_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
                input_feature_test_arr=preprocessor_obj.transform(input_features_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(train_target_feature)]
                test_arr = np.c_[input_feature_test_arr, np.array(test_target_feature)]
                logging.info("Saving Preprocessing Object")


                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_path,
                    obj=preprocessor_obj)
                

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_path
                )

            except Exception as e:
                    raise CustomException(e,sys)
            

            

        



