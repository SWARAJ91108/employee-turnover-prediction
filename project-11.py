import pandas as pd
# Load The dataset

df = pd.read_csv("employee_turnover.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())


#drop y column which we will predict 
X = df.drop("Employee_Turnover",axis=1)
y = df["Employee_Turnover"]

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
print("Model Trained Succesfully")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))




#L1 model ridge regression 

l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
l1_model.fit(X_train, y_train)
l1_pred = l1_model.predict(X_test)


l2_model = LogisticRegression(penalty='l2', random_state=42)
l2_model.fit(X_train, y_train)
l2_pred = l2_model.predict(X_test)


print("Baseline Accuracy:", accuracy)
print("L1 Accuracy:", accuracy_score(y_test, l1_pred))
print("L2 Accuracy:", accuracy_score(y_test, l2_pred))