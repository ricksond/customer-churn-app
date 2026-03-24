from preprocess import load_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import joblib
from sklearn.pipeline import Pipeline

# Load the data
df=load_df()

# Select Independent and dependent variables
X=df.drop("churn",axis=1)
y=df["churn"]

# Split into 80% train and 20% temp, then split temp into 10% validation and 10% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Create a pipeline with standard scaler and logistic regression 
pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
    ])

# Train the model
pipeline.fit(X_train, y_train)
print("Model trained successfully.")
print(pipeline)



