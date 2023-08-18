import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as tree
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/food_usa.csv') 
print(data)

print(data.isnull().sum())

y = data['Price 2023']
x = data.drop(['Product','Price 2023','Currency','Country'],axis=1)

#sns.pairplot(data)


#Linear reggresion
def calculateLinearReggression(x,y):
    model = LinearRegression()
    #60:40
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.4)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    #print(y_pred)
    print('Accuracy of linear regression classifier on test set: {:.3f}'.format(r2_score(Y_test, y_pred)))
    # A value of 1 indicates that predictions are identical to the observed values;


def calculateDecisionTreeRegression(x,y):
    #60:40
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.4)

    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, Y_train)

    y_pred = regressor.predict(X_test)
    print("Accuracy of decision tree regressior on test set: {:.3f}".format(regressor.score(X_test, Y_test)))

    
    # plt.figure(figsize=(10,10))
    # tree.plot_tree(regressor, feature_names = X_train.columns,  
    #            max_depth=5, filled=True);



calculateLinearReggression(x,y)
calculateDecisionTreeRegression(x,y)