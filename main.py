import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


task3_data = pd.read_csv('Task3 - dataset - HIV RVG.csv')
mean = task3_data[['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']].mean()
print(mean)
std = task3_data[['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']].std()
print(std)
min = task3_data[['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']].min()
print(min)
max = task3_data[['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']].max()
print(max)



target_column = ['Participant Condition']
predictors = ['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']
task3_data[predictors] = task3_data[predictors]/task3_data[predictors].max()
X = task3_data[predictors].values
y = task3_data[target_column].values
train, test = train_test_split(X, y, test_size=0.10, train_size=0.90)
mlp = MLPClassifier(hidden_layer_sizes=(500,500))
mlp.fit(train)
predict_train = mlp.predict(test)
#predict_test = mlp.predict(X_test)

