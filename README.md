# Data-Classification
To implement the logistic regression algorithm for classifying data.

## Dataset:
The dataset consists of three classes of points, with each point labeled as 0, 1, or 2. 
Each point has 4 features, (x1,x2,x3,x4). 

## Used two different ways of implementation:
1.	Scikit-Learn library to classify the data. 
2.	One-vs-Rest strategy in predicting the class of a data point.

For each of the two implementations, split the dataset into two datasets: training and testing datasets. Used the training dataset to develop the logistic regression model, then validate the model using the testing dataset. 

## Output:
1. W vector for each of the three classes
2. Plot of the cost as a function of the number of iterations, while computing the W vector for each class.
3. The accuracy in the predictions for each of the three binary classifiers.
4. The accuracy of the overall model, in predicting the testing dataset.
