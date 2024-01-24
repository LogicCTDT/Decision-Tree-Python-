# Visual Decision Tree Based on Categorical Attributes 
-------------------

As you may know "scikit-learn" library in python is not able to make a decision tree based on categorical data, and you have to convert categorical data to numerical before passing them to the classifier method. Also, the resulted decision tree is a binary tree while a decision tree does not need to be binary.

Here, we provide a library which is able to make a visual decision tree based on categorical data. You can read more about decision trees [here](https://en.wikipedia.org/wiki/Decision_tree).

## Features
--------------------

The main algorithm which is used is ID3 with the following features:

* Information gain based on [entropy](https://en.wikipedia.org/wiki/Decision_tree_learning)
* Information gain based on [gini](https://en.wikipedia.org/wiki/Decision_tree_learning)
* Some pruning capabilities like:
	* Minimum number of samples
	* Minimum information gain
* The resulted tree is not binary

## Requirements
--------------------

You can find all the requirements in "requirements.txt" file, and it can be installed easily by the following command:

* pip install -r requirements.txt 

Also to be able to see visual tree, you need to install graphviz package. [Here](https://www.graphviz.org/download/) you can find the right package with respect to your operation system. 


## Usage
--------------------

```python

from p_decision_tree.DecisionTree import DecisionTree
import pandas as pd

#Reading CSV file as data set by Pandas
raw_data = [[1, 'rain', 'strong', 'hot', 'no'],
            [2, 'overcast', 'weak', 'hot', 'yes'],
            [3, 'sunny', 'strong', 'cool', 'yes'],
            [4, 'overcast', 'strong', 'cool', 'no'],
            [5, 'rain', 'strong', 'hot', 'cool'],
            [6, 'overcast', 'strong', 'hot', 'no'],
            [7, 'sunny', 'strong', 'hot', 'no'],
            [8, 'overcast', 'weak', 'hot', 'no'],
            [9, 'sunny', 'weak', 'hot', 'no'],
            [10, 'rain', 'weak', 'hot', 'no'],
            [11, 'rain', 'weak', 'cool', 'no'],
            [12, 'sunny', 'weak', 'cool', 'yes'],
            [13, 'overcast', 'weak', 'hot', 'no']]

data = pd.DataFrame(raw_data, columns=['Datapoint', 'Overcast', 'Wind', 'Temp', 'Tennis?'])

columns = data.columns

#All columns except the last one are descriptive by default
descriptive_features = columns[:-1]
#The last column is considered as label
label = columns[-1]

#Converting all the columns to string
for column in columns:
    data[column]= data[column].astype(str)

data_descriptive = data[descriptive_features].values
data_label = data[label].values

#Calling DecisionTree constructor (the last parameter is criterion which can also be "gini")
decisionTree = DecisionTree(data_descriptive.tolist(), descriptive_features.tolist(), data_label.tolist(), "entropy")

#Here you can pass pruning features (gain_threshold and minimum_samples)
decisionTree.id3(0,0)

#Visualizing decision tree by Graphviz
dot = decisionTree.print_visualTree( render=True )

# When using Jupyter
#display( dot )

print("System entropy: ", format(decisionTree.entropy))
print("System gini: ", format(decisionTree.gini))



``` 

