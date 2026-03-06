from preprocess import load_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
df=load_df()

# Select Independent and dependent variables
X=df.drop("churn",axis=1)
y=df["churn"]

# Create a train_test_split with 80% training data and 20% testing data.
