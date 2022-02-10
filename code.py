import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

train_raw_df = fetch_20newsgroups(subset='train', categories=categories)
test_raw_df = fetch_20newsgroups(subset='test', categories=categories)

x_train, x_val, y_train, y_val = train_test_split(np.array(train_raw_df.data), train_raw_df.target, test_size=0.1)
x_test = np.array(test_raw_df.data)
y_test = test_raw_df.target

# x_train = [x_train[:200] for x in x_train]

print('Train:', len(x_train))
print('Val:', len(x_val))
print('Test:', len(x_test))