import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from prettytable import PrettyTable

def Standardize(df, num_col):
    for i in num_col:
        print(i)
        df[i] = (df[i] - df[i].mean())/df[i].std()
    return df

X_cook = [
    [1,0,0,0],
    [1,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
]

y_cook = [
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0
]

print("Conditionally dependent dataset in Naive Bayes")
clf = GaussianNB()
clf.fit(X_cook, y_cook)
predictions = clf.predict(X_cook)
for i in range(len(predictions)):
    print(f"Input: {X_cook[i]}, Predicted Output: {predictions[i]}")
print(f"Accuracy: {metrics.accuracy_score(y_cook,predictions)}")

# XOR input and output data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

clf = GaussianNB()
clf.fit(X, y)
predictions = clf.predict(X)
print('Xor problem using Naive Bayes')
for i in range(4):
    print(f"Input: {X[i]}, Predicted Output: {predictions[i]}")
print(f"Accuracy: {metrics.accuracy_score(y,predictions)}")

columns=['age','workclass','fnlwgt','education','education_num','marital-status','occupation','relationship','race',
         'sex','capital-gain','capital-loss','hours-per-week','native-country','income']

df=pd.read_csv(r"C:\Users\SrikarUmmineni\PycharmProjects\pythonProject\Urban_Computing\census+income\adult.data",header=None)
df_test = pd.read_csv(r"C:\Users\SrikarUmmineni\PycharmProjects\pythonProject\Urban_Computing\census+income\adult.test",header=None)
df.columns=columns
df_test.columns=columns
print(df.dtypes)
income_map = {' <=50K': 0, ' >50K': 1, ' <=50K.':0,' >50K.':1}
df['income'] = df['income'].map(income_map)
df_test['income'] = df_test['income'].map(income_map)
df=df.drop_duplicates(keep='first')
print("After dropping duplicates, the shape of the dataset is :", df.shape)

df=df.applymap(lambda x: np.nan if x==' ?' else x)
df_test=df_test.applymap(lambda x: np.nan if x==' ?' else x)

df=df.dropna()
df_test=df_test.dropna()

print("After dropping '?' from the training dataset, the number of null values or '?' are")
print(df.isna().sum().sum())
print(df[df=='?'].sum().sum())

df.drop(['capital-gain', 'capital-loss', 'native-country'], axis = 1, inplace = True)
df_test.drop(['capital-gain', 'capital-loss', 'native-country'], axis = 1, inplace = True)
df['education_num']=df['education_num'].astype('int8')

input_features      = [col for col in df.columns if not col in ['education','income']]
numerical_columns   = ['age', 'fnlwgt','hours-per-week','education_num']
categorical_columns = [col for col in input_features if not col in numerical_columns]

df = pd.get_dummies(df, columns = categorical_columns, drop_first=True)
df_test = pd.get_dummies(df_test, columns = categorical_columns, drop_first=True)

df=Standardize(df, numerical_columns)
df_test=Standardize(df_test, numerical_columns)

df.drop('education', axis = 1, inplace = True)
df_test.drop('education', axis = 1, inplace = True)

X = df.drop('income', axis=1).values
y = df['income'].values

perceptron = Perceptron()

# # Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracy= []

# # Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracy.append(accuracy)
    # Calculate precision, recall, and F1 score for this fold

X_test = df_test.drop('income', axis=1).values
y_test = df_test['income'].values
y_pred_test = perceptron.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
print(f"Accuracy of single Perceptron: {accuracy_test}")
print(f"Precision of single Perceptron: {precision_test}")
print(f"Recall of single Perceptron: {recall_test}")
print(f"F1 score of single perceptron: {f1_test}")

average_accuracy = np.mean(fold_accuracy)

print("Accuracy for each fold for single Perceptron for train set :", fold_accuracy)
print("Average Accuracy for single Perceptron for train set:", average_accuracy)

pt = PrettyTable()
pt.field_names = ['Model', 'Accuracy score', 'Precision score', 'Recall score', 'F1 score']
pt.add_row(['Single Perceptron', accuracy_test, precision_test, recall_test, f1_test])

mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracy = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracy.append(accuracy)

average_accuracy = np.mean(fold_accuracy)

print("Accuracy for each fold for Multilayer Perceptron for train set:", fold_accuracy)
print("Average Accuracy for Multilayer Perceptron for train set:", average_accuracy)

X_test = df_test.drop('income', axis=1).values
y_test = df_test['income'].values
y_pred_test = mlp.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
print(f"Accuracy of Multilayer Perceptron with test set: {accuracy_test}")
print(f"Precision of Multilayer Perceptron with test set: {precision_test}")
print(f"Recall of Multilayer Perceptron  with test set: {recall_test}")
print(f"F1 score of Multilayer perceptron: {f1_test}")

pt.add_row(['Multilayer Perceptron', accuracy_test, precision_test, recall_test, f1_test])
print(pt)
