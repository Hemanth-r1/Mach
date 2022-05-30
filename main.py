#  Create a classification model to predict the gender (male or female) based on different acoustic parameters
# The Dataset I am using contains data for identifying male and female voice based on different acoustic properties.
# Importing the basic python libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# importing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# importing the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# importing K Neighbour Classifier
from sklearn.neighbors import KNeighborsClassifier
# importing Linear Regression Model
from sklearn.linear_model import LogisticRegression
# importing SVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

# Importing Data
df = pd.read_csv('voice.csv')
df.head()

# Descriptive Analysis of the Dataset
print("Shape of Gender Voice Recognition is", df.shape)
df.info()

# Various factors important by statistical means like mean, standard deviation, median, count of values and maximum
# value etc. are shown below for numerical variables of our dataset.

df.describe()

# Seeing what are the columns present in the dataset
df.columns

# There are 21 columns in this dataset, so it's important to check whether
# this dataset contains null values before going any further.

# Checking for Null values
df.isnull().sum()
# From above, we can see that it have no null values.

# Processing Categorical Values
# Here we will perform encoding. Encoding by converting 'label' feature into numerical form.

lb = LabelEncoder()
df['label'] = lb.fit_transform(df['label'])
df.head()

# Visualization of the Dataset
# The **label** column in this dataset is what we have predict.
# So let's see the distribution of the values of the *label* column.

d1 = df['label'].value_counts()
d1
# 1 is Male
# 0 is Female

plt.pie(d1, labels=['Male', 'Female'], autopct="%.2f%%")
plt.title('Male and Female label classification')
plt.show()

# From above, we get the visual representation with comparison between **male** and **female**.
# We can also see that our data is perfectly balanced. <br>
# Now let's have a look at the correlation among the dataset:

print(df.corr())

plt.figure(figsize=(15, 10))
correalation = df.corr()
sns.heatmap(correalation, cmap='coolwarm', annot=True)
plt.show()

# Selecting Features
# Here we are selecting the independent and dependent variables for the training of the dataset.
# Selecting the independent variables(features) from the dataset
x = df.iloc[:, :-1]
x.head(3)
type(x)

# Selecting the dependent variable from the dataset
y = df['label']
y.head()
type(y)

# Train-test Splitting
# Here I am spliting the data into training and test sets and using
# different Machine Learning algorithms to train the model.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Applying Classifier Models
# Before applying model we will create a function which will take
# input a model and help us to reuse the code again and again.

def algo_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('Prediction')
    print(pred)
    print("Training score: ", model.score(x_train, y_train))
    print("Testing Score: ", model.score(x_test, y_test))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, pred))
    print('Classification Report')
    print(classification_report(y_test, pred))


# a.Decision Tree Classifier

model_1 = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=20)

algo_model(x_train, x_test, y_train, y_test, model_1)

# b.Random Forest Classifier

model_2 = RandomForestClassifier(n_estimators=75, criterion='entropy')

algo_model(x_train, x_test, y_train, y_test, model_2)

# c.KNN Classifier

model_3 = KNeighborsClassifier(n_neighbors=27)

algo_model(x_train, x_test, y_train, y_test, model_3)

# d.Logistic Regression

model_4 = LogisticRegression(solver='liblinear')

algo_model(x_train, x_test, y_train, y_test, model_4)

# SVM Classifier

# using kernal as 'linear'

model_5 = SVC(kernel='linear', C=10)
algo_model(x_train, x_test, y_train, y_test, model_5)

# using kernal as 'rbf'

model_6 = SVC(kernel='rbf', C=10)
algo_model(x_train, x_test, y_train, y_test, model_6)

# using kernal as 'poly'

model_7 = SVC(kernel='poly', C=10)
algo_model(x_train, x_test, y_train, y_test, model_7)

# Conclusion
#
# 1. We can see from using the **Decision Tree Classifier** that the Testing score and accuracy of the dataset is **0.9684542586750788** and **0.97** respectively.
# 2. We can see from using the **Random Forest Classifier** that the Testing score and accuracy of the dataset is **0.9826498422712934** and **0.98** respectively.
# 3. We can see from using the **KNN Classifier** that the Testing score and accuracy of the dataset is **0.7097791798107256** and **0.71** respectively.
# 4. We can see from using the **Logistic Regression** that the Testing score and accuracy of the dataset is **0.916403785488959** and **0.92** respectively.
# 5. We can see from using the **SVM classifier with kernel as 'linear'** that the Testing score and accuracy of the dataset is **0.9747634069400631** and **0.97** respectively.
# 6. We can see from using the **SVM classifier with kernel as 'rbf'** that the Testing score and accuracy of the dataset is **0.7018927444794952** and **0.70** respectively.
# 7. We can see from using the **SVM classifier with kernel as 'poly'** that the Testing score and accuracy of the dataset is **0.5220820189274448** and **0.52** respectively.
#
# From the above ML algorithms we can see that the model which provide us with the greatest accuracy is the **Random Forrest Classifier** with the accuracy of **98%**.

# Prediction on test data by using Random Forrest Classifier

model_2 = RandomForestClassifier(n_estimators=75, criterion='entropy')
model_2.fit(x_train, y_train)
pred = model_2.predict(x_test)
outputDF = pd.DataFrame({'Actual Label': y_test, 'Predicted Label': pred})
print(outputDF.head(20))

# Finally from above we can see that the **Actual Label** and **Predicted Label** have same labels determining the high accuracy.

# Saving the model

filename = 'final_model.sav'
pickle.dump(model_2, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result, '% Accuracy')