import os
import pandas
 
#Changing the current working directory
os.chdir("D:/Ediwsor_Project - Bike_Rental_Count")
BIKE = pandas.read_csv("Bike.csv")
bike = BIKE.copy()
 
categorical_col_updated = ['season','yr','mnth','weathersit','holiday']
bike = pandas.get_dummies(bike, columns = categorical_col_updated) 
 
#Separating the dependent and independent data variables into two data frames.
from sklearn.model_selection import train_test_split 
 
X = bike.drop(['cnt'],axis=1) 
Y = bike['cnt']
 
# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)
