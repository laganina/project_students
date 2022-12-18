import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#sep is separator
data = pd.read_csv('student-mat.csv', sep=';')

#trimming data
#attributes
data = data[['G1','G2','G3','studytime','failures','absences']]

predict = 'G3'

#returns new dataframe
#based on this we predict another value
x = np.array(data.drop([predict],1))
y = np.array(data[predict]) #we only care about the actual G3 value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

''''
best = 0
for _ in range(30):
#splitting in four var

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
#x_train is section of x, y_train is section of y
#we're splitting 10% of data into test results

#training model

    linear = linear_model.LinearRegression()

#fitting data to fit to find the best fit line
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
#skipped training process
        best = accuracy
        with open('studentmodel.pickle','wb') as f:
            pickle.dump(linear, f)'''

pickle_in = open('studentmodel.pickle', 'rb')

linear = pickle.load(pickle_in)

#what are b and n
#intercept is y
#n coeficence
print('Co:\n', + linear.coef_)
#shows us the y intercept
#a line in 5 dimensional space
print('Intercept \n', + linear.intercept_)

#using on real students
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
#predicts close to the real answer

#saving models, plotting and visualising data

p = 'absences'
style.use('ggplot')
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()



