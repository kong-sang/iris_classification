from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
#print(iris)

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df['target'] = df['target'].map({0:"setosa", 1:"versicolor", 2:"virginica"})

df.to_csv("iris.csv", "w")
