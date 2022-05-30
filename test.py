import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('voice.csv')
df.head()
print("Shape of Gender Voice Recognition is", df.shape)
df.info()

df.describe()
# Seeing what are the columns present in the dataset
df.columns
df.isnull().sum()
lb = LabelEncoder()
df['label'] = lb.fit_transform(df['label'])
df.head()

d1 = df['label'].value_counts()
d1
# 1 is Male
# 0 is Female

plt.pie(d1, labels=['Male', 'Female'], autopct="%.2f%%")
plt.title('Male and Female label classification')
plt.show()

print(df.corr())

plt.figure(figsize=(15, 10))
correalation = df.corr()
sns.heatmap(correalation, cmap='coolwarm', annot=True)
plt.show()

# Selecting the independent variables(features) from the dataset
x = df.iloc[:, :-1]
x.head(3)
type(x)

# Selecting the dependent variable from the dataset
y = df['label']
y.head()
type(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# applying classifier model
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

    # importing Decison Tree Classifier
    from sklearn.tree import DecisionTreeClassifier

    model_1 = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=20)

    algo_model(x_train, x_test, y_train, y_test, model_1)

# importing the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model_2 = RandomForestClassifier(n_estimators=75, criterion='entropy')


algo_model(x_train, x_test, y_train, y_test, model_2)

# importing K Neighbour Classifier
from sklearn.neighbors import KNeighborsClassifier

model_3 = KNeighborsClassifier(n_neighbors=27)

algo_model(x_train, x_test, y_train, y_test, model_3)

# importing Linear Regression Model
from sklearn.linear_model import LogisticRegression

model_4 = LogisticRegression(solver = 'liblinear')

algo_model(x_train, x_test, y_train, y_test, model_4)

# importing SVC
from sklearn.svm import SVC

model_5 = SVC(kernel = 'linear', C = 10)
algo_model(x_train, x_test, y_train, y_test, model_5)

model_6 = SVC(kernel = 'rbf', C = 10)
algo_model(x_train, x_test, y_train, y_test, model_6)

model_7 = SVC(kernel = 'poly', C = 10)
algo_model(x_train, x_test, y_train, y_test, model_7)

#Conclusion
#We can see from using the Decision Tree Classifier that the Testing score and accuracy of the dataset is 0.9684542586750788 and 0.97 respectively.
##We can see from using the Random Forest Classifier that the Testing score and accuracy of the dataset is 0.9826498422712934 and 0.98 respectively.
#We can see from using the KNN Classifier that the Testing score and accuracy of the dataset is 0.7097791798107256 and 0.71 respectively.
#We can see from using the Logistic Regression that the Testing score and accuracy of the dataset is 0.916403785488959 and 0.92 respectively.
#We can see from using the SVM classifier with kernel as 'linear' that the Testing score and accuracy of the dataset is 0.9747634069400631 and 0.97 respectively.
#We can see from using the SVM classifier with kernel as 'rbf' that the Testing score and accuracy of the dataset is 0.7018927444794952 and 0.70 respectively.
#We can see from using the SVM classifier with kernel as 'poly' that the Testing score and accuracy of the dataset is 0.5220820189274448 and 0.52 respectively.
#From the above ML algorithms we can see that the model which provide us with the greatest accuracy is the Random Forrest Classifier with the accuracy of 98%.


from sklearn.ensemble import RandomForestClassifier
model_2 = RandomForestClassifier(n_estimators=75, criterion='entropy')

model_2.fit(x_train, y_train)
pred = model_2.predict(x_test)
outputDF = pd.DataFrame({'Actual Label': y_test, 'Predicted Label': pred})
print(outputDF.head(20))

# Saving the model
import pickle
filename = 'final_model.sav'
pickle.dump(model_2, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result, '% Accuracy')