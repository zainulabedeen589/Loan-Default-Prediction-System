import pandas as pd
from src.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import sys
import numpy as np  
import os
# print("DATA PROCESSING PATH : ",PROCESSED_DATA_PATH)
# os.makedirs(PROCESSED_DATA_PATH , exist_ok=True)
logger = get_logger(__name__)

class DataProcessor:

    def __init__(self):
        self.train_path = TRAIN_DATA_PATH
        self.processed_data_path = PROCESSED_DATA_PATH

    def load_data(self):
        try:
            logger.info("Data Processinng Started")
            df = pd.read_csv(self.train_path)
            for col in df.columns:
                # Try to convert each column; if it fails, it stays as is
                try:
                    df[col] = df[col].astype(float) # pd.to_numeric(df[col], errors='ignore')
                    
                except:
                    pass

            logger.info(f"Data read succesfull : Data shape : {df.shape}")
            return df
        except Exception as e:
            logger.error("Problem while Loading Data")
            raise CustomException("Error while loading data : ",sys)
        
    def drop_unnecessary_columns(self, df, columns):
        try:
            logger.info(f"Dropping Unnecesary Columns :  {columns}")
            # df = df.drop(columns = columns, axis=1)
            df = df.drop(columns = columns)
            logger.info(f"Columns dropped Sucesfully : Shape = {df.shape}")
            return df
        except Exception as e:
            logger.error("Problem while dropping columns")
            raise CustomException("Error while dropping columns : ",sys)
    
    def handle_outliers(self, df , columns):
        try:
            logger.info(f"Handling outliers : Columns = {columns}")
            for column in columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Outliers handled  Sucesffuully : {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Problem while Outlier handling")
            raise CustomException("Error while outlier handling : ",sys)
    
    def handle_null_values(self,df,columns):
        try:
            logger.info("Handling null values")
            df[columns] = df[columns].fillna(df[columns].mean())
            logger.info(f"Missing values handled sucessfully : Shape == {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Problem while null values handling")
            raise CustomException("Error while null values handling : ",sys)
        
    def save_data(self,df):
        try:
            os.makedirs(PROCESSED_DIR , exist_ok=True)
            df.to_csv(self.processed_data_path, index=False)
            logger.info("Processed data saved sucesfully")
        
        except Exception as e:
            logger.error("Problem while saving data")
            raise CustomException("Error while saving data : ",sys)
    
    def run(self):
        try:
            logger.info("Starting the pipeline of Data Procesing")

            df = self.load_data()
            df = self.drop_unnecessary_columns(df , ["LoanID"])
            columns_to_handel= num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
            
            # ['LoanAmount','InterestRate','LoanTerm','Monthly_Payment','PTI_Ratio','Job_Stability_Index','Debt_per_Line','Young_High_Risk']
            df = self.handle_outliers(df , columns_to_handel)

            # df = self.handle_null_values(df , 'Monthly_Payment')

            self.save_data(df)

            logger.info("Data Proccesing Pipeline Completed Sucessfully")

        except CustomException as ce:
            logger.error(f"Error ocuured in Data Processing Pipleine : {str(ce)}")

if __name__=="__main__":
    processor = DataProcessor()
    processor.run()




        
        
        
    