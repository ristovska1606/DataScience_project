import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv('data/food_usa.csv') 
print(data)

print(data.isnull().sum())

sns.pairplot(data)

#Linear reggresion
model = LinearRegression()
y = data['Price 2023']
x = data.drop(['Product','Price 2023','Currency','Country'],axis=1)
#60:40
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.4)
print("Training data: " + X_train.shape)
print("Testing data: " + X_test.shape)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(y_pred)
print('Accuracy of linear regression classifier on test set: {:.3f}'.format(r2_score(Y_test, y_pred)))
# A value of 1 indicates that predictions are identical to the observed values;
