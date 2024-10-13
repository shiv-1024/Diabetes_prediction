import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
sns.set_theme()

diabetes_df = pd.read_csv('diabetes.csv')

print(diabetes_df.head())
print(f'The Shape of the dataset : {diabetes_df.shape}')
print(diabetes_df.info())
print(diabetes_df.isnull().sum())
print(diabetes_df.describe())

print(diabetes_df.value_counts('Outcome'))
print(diabetes_df.groupby('Outcome').mean())

print(f'The Duplicates value of the dataset is {diabetes_df.duplicated().sum()}')

#Data Visualization
plt.figure(figsize = (6,6))
sns.countplot(data = diabetes_df , x ='Outcome')
plt.show()

#Outliers
plt.figure(figsize = (12,12))
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=col,data =diabetes_df)
plt.tight_layout()
plt.show()

plt.figure(figsize = (12,12))
for i,col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    plt.subplot(3,3,i+1)
    sns.histplot(data = diabetes_df , x = col , kde = True)
plt.tight_layout()
plt.show()

sns.pairplot(data = diabetes_df, hue = 'Outcome')
plt.show()

plt.figure(figsize=(12,6))
sns.heatmap(diabetes_df.corr(),cbar = True ,square = True ,fmt = '.1f',annot = True,annot_kws = {'size' :8},cmap = 'Blues')

print(diabetes_df.std())

#Data Standaridization
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(diabetes_df.drop(['Outcome'] ,axis = 1)),columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y = diabetes_df['Outcome']
x.head()

#splitting into trainn and test dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y,random_state =4)

#Training Model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)

#The Predicticting Accuracy Score For Training Data 
x_training_prediction =classifier.predict(x_train)
training_data_accuracy =accuracy_score(x_training_prediction,y_train)
print(f'The Accuracy Score of the training data :{round(training_data_accuracy*100)}%')

#The Predicticting Accuracy Score For Testing Data 
x_training_prediction =classifier.predict(x_test)
testing_data_accuracy =accuracy_score(x_training_prediction,y_test)
print(f'The Accuracy Score of the testing data :{round(testing_data_accuracy*100)}%')

#Making Predicting System 
input_data = [10,168,74,0,0,38,0.537,34]
input_data_into_array = np.asarray(input_data)
input_data_reshape = input_data_into_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshape)

if prediction[0] == '0':
    print("The person doesn't have diabetes")
else:
    print(f'The Person has diabetes')