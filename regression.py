#importing libraries
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


#collecting dataset
dataset = pd.read_csv("task-1.csv")


#'y' value of dataset(target)
target = dataset["Temp. (Degree Celcius)"]
#target


#slicing day from the Time column of dataset(dataset1)
start, stop, step = 0, -42, 1
dataset1 = dataset["Time"].str.slice(start , stop, step)


#slicing month from the Time column from dataset(dataset2)
start, stop, step = 3, -38, 1
dataset2 = dataset["Time"].str.slice(start, stop, step)

#replacing month by its integer number(jan-1,feb-2 etc)
dataset2 = dataset2.str.replace('Jan', '1').str.replace('Feb', '2').str.replace('Mar', '3').str.replace('Apr', '4').str.replace('May', '5').str.replace('Jun', '6').str.replace('Jul', '7').str.replace('Aug', '8').str.replace('Sep', '9').str.replace('Oct', '10').str.replace('Nov', '11').str.replace('Dec', '12')


#slicing year from the Time column from dataset(dataset3)
start, stop, step = 7, -32, 1
dataset3 = dataset["Time"].str.slice(start, stop, step)


#slicing initial time from the Time column from dataset(start_time)
start, stop, step = 12, -30,1
start_time = dataset["Time"].str.slice(start, stop, step)
#start_time


#slicing am/pm value from 'Time' column from dataset(am)
start, stop, step = 18, -24,1
am = dataset["Time"].str.slice(start, stop, step)

#replacing am by 1 and pm by 2
am = am.str.replace('am', '1').str.replace('pm', '2')
#am



#slicing final time from Time column from dataset(end_time)
start, stop, step = 36, -6,1
end_time = dataset["Time"].str.slice(start, stop, step)
#end_time



#slicing am/pm value from 'Time' column from dataset(am)
start, stop, step = 42, -1,1
pm = dataset["Time"].str.slice(start, stop, step)

#replacing am by 1 and pm by 2
pm = pm.str.replace('a', '1').str.replace('p', '2')
#pm


#concating all the individual data as an input dataset which consists of columns days,mnths,years,starting time,ending time,y value etc
ip_dataset = pd.concat([dataset1,dataset2,dataset3,start_time,am,end_time,pm,target], axis=1, ignore_index=True)
#ip_dataset


#renaming the columns of the input dataset
ip_dataset.columns = ['day', 'month', 'year', 'start_time', 'am/pm', 'end_time' ,'pm/am','temp']
#ip_dataset


#filling 'NA' values of input dataset to a random temperature 20.0
ip_dataset['temp'] = ip_dataset['temp'].fillna(20.0)


#plotting the dataset(independent variable vs dependent variable)
plt.scatter(range(20),ip_dataset['temp'][:20], color='red')
plt.title('features Vs label', fontsize=14)
plt.xlabel('start_time', fontsize=14)
plt.ylabel('temp', fontsize=14)
plt.grid(True)
plt.show()


#taking X as feature vector and Y as target for trainig the model
X = ip_dataset[['day','month','year','start_time','am/pm',]]
Y = ip_dataset['temp']


#split the dataset into train and test with 80:20 ratio
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 



#linear regression model 
regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)


#printing the weights of the hypoyhesis ie b0,b1,b2,b3,b4,b5
print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)
Y_pred = regressor.predict(X_train)
plt.scatter(range(20),Y_train[:20], color='red')
plt.scatter(range(20),Y_pred[:20],color = 'blue')
plt.title('features Vs label', fontsize=14)
plt.xlabel('start_time', fontsize=14)
plt.ylabel('temp', fontsize=14)
plt.grid(True)
plt.show()
#test the model with test dataset



#comparing actual y value with predicted y value
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df 


#printing mean squared error for linear regression model
from sklearn.metrics import mean_squared_error, r2_score
print("R2 score : %.2f" % np.sqrt(mean_squared_error(Y_test, Y_pred)))#2.93


#another polynommial regression model for better accuracy with degree 4
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)


#test the model with test dataset
y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
plt.scatter(range(20),Y_test[:20], color='red')
plt.scatter(range(20),y_test_predict[:20],color = 'blue')
plt.title('features Vs label', fontsize=14)
plt.xlabel('start_time', fontsize=14)
plt.ylabel('temp', fontsize=14)
plt.grid(True)
plt.show()

#comparing actual y value with predicted y value
df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': y_test_predict})  
df1


#evaluating the model on test dataset
rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
r2_test = r2_score(Y_test, y_test_predict)

#printing mean squared error and r-square error for polynomial regression model
print(rmse_test)#2.14
print(r2_score)