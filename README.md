# Spam-Ham-mail-detection

Name : Pranav Subhash Tikone
Project no. : 2
Project Name : SPAM / HAM mail detection
Date : 20/04/23
Topic : Binary Classification ( Supervised ML )

Description :

This project classifies the SPAM and HAM mails and determines whether the input mail is SPAM or HAM.

BINARY CLASSIFICATION is a model which classifies the provided data into 2 categories with output values either 0 or 1.
In this project the 0 label refers to the HAM mails and 1 label refers to the SPAM mails.

The mathematical function used for this type of classification is SIGMOID function. This output values of this function ranges from 0 to 1, thus providing us the output values in between this range; [0,1]

To classify the things that whether it falls in class 0 or 1, DECISION BOUNDARY provides us a basic idea of how the classified data looks like with respect to this boundary. This is used to divide the outputs into the two classes with respect to this decision boundary.

The model used for this binary classification task is LOGISTIC REGRESSION. This uses sigmoid function and updates the values of the paramaters using the GRADIENT DESCENT ALGORITHM after each iteration. The updated parameters are finalized with a value which minimizes the COST and LOSS function.

The modules used for this binary classification are :

1. PANDAS : To load the dataset of the spam and ham mails.
2. SCIKIT LEARN : 
   
   a) To import the LOGISTIC REGRESSION ALGORITHM
   b) To Split the available data into TRAINING and TESTINNG DATASETS
   c) To test the ACCURACY SCORE of your model
   d) To convert the mails's TEXT to NUMBERS (using Tfidfvectorizer) for computing the output usinng logistic regression

The original dataset is divided into the training and test data sets with training data being 80% of the original dataset and test data being 20% of the original dataset.

I have pasted one of the mails recieved from LEETCODE to me, to check whether it is SPAM or HAM. ( Ofcourse its a HAM)
The model correctly predicts the mail as a HAM mail and provides us with the required output.
You can also try pasting the mails you recieved in the INPUT MAIL between " " and run the code to check if the mail you put as input is a HAM / SPAM accordingly.
NOTE: IF YOUR MAIL IS A MULTILINE TEXT, USE 3 INVERTED COMMAS FOR START AND END AND PASTE YOUR MAIL BETWEEN THESE [""" YOUR INPUT MAIL """]

*** To look through all the MENTIONED DATA, SHAPES OF MATRICES,TEXT MAILS TRANSFORMED INTO NUMBERS etc. ,  remove the Hashtags before the statements which you want to compute and run the code ***  
