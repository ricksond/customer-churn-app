import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv();
#loading the Database url from env for security
DATABASE_URL=os.getenv("DATABASE_URL")

engine=create_engine(DATABASE_URL)

df=pd.read_csv("data/telco-customer-churn.csv")

df.to_sql("customers",engine,if_exists="append",index=False)

print("Data printed into postgres")