import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from preprocess import load_df
from preprocess import load_df

def train_xgb_model(): 
    df=load_df()

    # Select indepedent and dependent variables
    X=df.drop("churn",axis=1)
    y=df["churn"]

    # Split into 80% train and 20% temp, then split temp into validation and test sets.
    X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

    print(len(X_train),len(X_val),len(X_test))

    model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
    )

    model=model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=True)
    print(model)
    print("Model Trained Successfully")

    # Validation performance
    val_predictions=model.predict(X_val)
    val_probabilities=model.predict_proba(X_val)[:,1]

    val_auc=roc_auc_score(y_val,val_probabilities)
    val_acc=accuracy_score(y_val,val_probabilities.round())

    print(f"Validation AUC: {val_auc}")
    print(f"Validation Accuracy: {val_acc}")

    # roc_auc_score and accuracy on test set
    test_prediction=model.predict(X_test)
    test_proba=model.predict_proba(X_test)[:,1]
    test_auc=roc_auc_score(y_test,test_proba)
    test_acc=accuracy_score(y_test,test_prediction)
    print(f"Test AUC: {test_auc}")
    print(f"Test Accuracy: {test_acc}")
    model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"xgb_churn_model.joblib")
    joblib.dump(model,model_path)
    return model




if __name__ == "__main__":
    train_xgb_model()