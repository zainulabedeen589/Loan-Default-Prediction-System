import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from src.logger import get_logger
from src.custom_exception import CustomException
from src.paths_config import *
from utils.helpers import *
import sys

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.data_path = PROCESSED_DATA_PATH
        # self.df = None
        self.label_mapping={}

    def load_data(self):
        try:
            logger.info("Loading data")
            self.df = pd.read_csv(self.data_path)
            logger.info("Data loaded sucesffuly")
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Error while loading data" , sys)
        
    def feature_construction(self):
        try:
            logger.info("Starting Feature Construction")
            self.total_payback = self.df['LoanAmount'] * (1 + (df['InterestRate'] / 100))
            self.df['Monthly_Payment'] = self.total_payback / df['LoanTerm']
            self.df['PTI_Ratio'] = self.df['Monthly_Payment'] / (self.df['Income'] / 12)
            self.df['Job_Stability_Index'] = self.df['MonthsEmployed'] / (self.df['Age'] * 12)
            self.df['Debt_per_Line'] = self.df['LoanAmount'] / (self.df['NumCreditLines'] + 1)
            self.df['Young_High_Risk'] = ((self.df['Age'] < 30) & (self.df['LoanAmount'] > self.df['LoanAmount'].median())).astype(int)

            logger.info("Feature Construction done sucesfully")

        except Exception as e:
            logger.error(f"Error while feature construction {e}")
            raise CustomException("Error while faeture construction" , sys)
    
    def bin_age(self):
        try:
            logger.info("Starting Binning of Age Column")
            self.df['Age_Group'] = pd.cut(self.df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Senior', "Super Senior"])
            self.df=self.df.drop(columns=['Age'])
            logger.info(f"Columns after binning : {self.df.columns}")
            logger.info("Binning of Age Column Sucesfull")
        except Exception as e:
            logger.error(f"Error while binning {e}")
            raise CustomException("Error while binning" , sys)
        
    def label_encoding(self):
        try:
            columns_to_encode = ['Age_Group', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
            logger.info(f"Performing label encoding on {columns_to_encode}")

            self.df , self.label_mapping=label_encode(self.df , columns_to_encode)

            for col,mapping in self.label_mapping.items():
                logger.info(f"Mapping for {col} : {mapping}")
            logger.info("Label Encoding sucesffull")

        except Exception as e:
            logger.error(f"Error while label encoding {e}")
            raise CustomException("Error while label encoding" , sys)


    def feature_selection(self):
        try:
            logger.info("Starting Feature Selection")
            X = self.df.drop(columns='satisfaction')
            y = self.df['satisfaction']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            mutual_info = mutual_info_classif(X_train, y_train, discrete_features=True)

            mutual_info_df = pd.DataFrame({
                                'Feature': X.columns,
                                'Mutual Information': mutual_info
                                }).sort_values(by='Mutual Information', ascending=False)
        
            logger.info(f"Mutual Information Table is : \n{mutual_info_df}")

            top_features = mutual_info_df.head(12)['Feature'].tolist()

            self.df = self.df[top_features + ['satisfaction']]
            logger.info(f"Top features : {top_features}")
            logger.info("Feature Selection Sucesfull")

        except Exception as e:
            logger.error(f"Error while feature slection {e}")
            raise CustomException("Error while feature selection" , sys)

    def save_data(self):
        try:
            logger.info("Saving your data ...")
            os.makedirs(ENGINEERED_DIR,exist_ok=True)
            self.df.to_csv(ENGINEERED_DATA_PATH,index=False)
            logger.info(f"Data Saved succesfully at {ENGINEERED_DATA_PATH}")

        except Exception as e:
            logger.error(f"Error while saving data {e}")
            raise CustomException("Error while saving data" , sys)
        
    def run(self):
        try:
            logger.info("Starting your feature enginnering pipeline")
            self.load_data()
            # self.feature_construction()
            self.bin_age()
            self.label_encoding()
            self.feature_construction()
            self.feature_selection()
            self.save_data()
            logger.info("Your FE pipeline sucesfully done..")

        except Exception as e:
            logger.error(f"Error in FE pipeline {e}")
            raise CustomException("Error in FE pipeline" , sys)
        
        finally:
            logger.info("End of FE pipeline")


if __name__ == "__main__":
    feature_engineer = FeatureEngineer()
    feature_engineer.run()
        
