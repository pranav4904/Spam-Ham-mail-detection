


# IDENTIFICATION OF SPAM AND HAM MAILS

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
 
# Loading the raw or initial mail data

initial_mail_data = pd.read_csv(r"C:\Users\ASUS\Desktop\Pranav\Machine Learning\Datasets\spam_ham_dataset.csv")
#print(initial_mail_data)

# Replacing Spaces with Null Strings
 
mail_data = initial_mail_data.where(pd.notnull(initial_mail_data),'')

# Separating the data into X (Input Mails) and Y (Spam or Ham mail labels) sets

# Label = 0 = HAM MAIL
# Label = 1 = SPAM MAIL

x = mail_data['text']
y = mail_data['label_num']

#print(f"X dataset = {x}")         # Printing the X data set
#print(f"Y dataset = {y}")         # Printing the Y data set

# Splitting the data sets into training and testing data sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

#print(f"X dataset shape = {x.shape}")                     # Printing the shape of the X dataset matrix
#print(f"X training set shape = {x_train.shape}")          # Printing the shape of X training set matrix
#print(f"X test set shape = {x_test.shape}")               # Printing the shape of X test set matrix

# Converting the text data into the numerical values to compute LOGISTIC REGRESSION output 

function = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=(True))

transformed_x_train = function.fit_transform(x_train)
transformed_x_test = function.transform(x_test)

#print(f"Transformed X train set = {transformed_x_train}")     # Printing the converted numerical values of X training set
#print(f"Transformed X test set = {transformed_x_test}")       # Printing the converted numerical values of X test set

# Training the datasets using LOGISTIC REGRESSION MODEL

model = LogisticRegression()

model.fit(transformed_x_train, y_train)
prediction_x_train_set = model.predict(transformed_x_train)

accuracy_score_train_set = accuracy_score(y_train, prediction_x_train_set)

#print(f"Accuracy score of Training set = {(accuracy_score_train_set*100):.2f} %")     # Accuracy score of the Training Set Prediction  

model.fit(transformed_x_test, y_test)
prediction_x_test_set = model.predict(transformed_x_test)

accuracy_score_test_set = accuracy_score(y_test, prediction_x_test_set)   

#print(f"Accuracy score of Test Set = {(accuracy_score_test_set*100):.2f} %")      # Accuracy score of the Test set Preddiction


# Implementing the code for real life examples for detection of SPAM and HAM for any INPUT MAIL
 
input_mail = [""]          # copy and paste the mail in the input mail an run the code to test SPAM or HAM

transformed_input_mail = function.transform(input_mail)
my_prediction = model.predict(transformed_input_mail)

if my_prediction[0] == 1:
    
    print("The mail is SPAM")
    
else:
    
    print("The mail is HAM")