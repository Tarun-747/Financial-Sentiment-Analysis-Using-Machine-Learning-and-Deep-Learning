import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/logistic_reg.pkl")
    return model

def evaluate_ml(model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()
