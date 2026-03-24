# customer-churn-app

STEP 1: Data Engineering Pipeline
1. Established a connection to a PostgreSQL database and loaded the raw Telco Customer Churn dataset into a table.
2. Identified the relevant columns for churn prediction and ensured each column was converted to the appropriate data type.
3. Removed or prevented duplicate records within the selected feature set.
4. Created a separate table containing only the cleaned and relevant features for modeling.
5. Exported the processed table to a CSV file for use in the machine learning pipeline.

Step 2: Machine Learning Pipeline
(To be implemented)

Planned steps:
1. Load the cleaned dataset. (Done)
2. Perform feature preprocessing and convert them into train test sets. (Done)
3. Train machine learning models to predict customer churn.
4. Evaluate model performance using appropriate metrics.
5. Save the trained model for API deployment.