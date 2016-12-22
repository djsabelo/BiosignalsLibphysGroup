import pandas
from pylab import*
import random
import _pickle as cPickle
from sklearn import linear_model, model_selection

results = pandas.read_csv("../data/table_features/features_OldMaximizer.csv", sep=";", index_col=0)
results = results[isfinite(results['vx_max'])]

y = results[results.columns[2]]

change_features = False
change_max = True
if change_features:
    print("Random features")
    temp_x = results[results.columns[8:]]
    for j in temp_x.columns.tolist():
        x = temp_x[j].tolist()
        for i in arange(0, len(temp_x[j])):
            temp_x.loc[temp_x.index.values[i], j] = random.gauss(mean(x), std(x))

if change_max:
    print("Random MAX")
    yy = y
    for i in arange(0, len(results['MAX'])):
        results.loc[results.index.values[i], 'MAX'] = random.gauss(mean(yy), std(yy))
    y = results[results.columns[2]]
    temp_x = results[results.columns[8:]]

features_name = temp_x.columns.values
features = array(temp_x)
x = array([features[i, :] for i in arange(0, len(y))])

loaded_model = cPickle.load(open('model.sav', 'rb'))
error = loaded_model.predict(x)
outliers = (len(find(error > 5)) + len(find(error < 0))) / len(error)
print('Use model: \n Score: %.2f' % loaded_model.score(x, y))
print('\n Residual sum of squares:' + str(mean((loaded_model.predict(x) - y) ** 2)))
print('\n Outliers:' + str(outliers))


regr = linear_model.LinearRegression()
regr.fit(x, y)
print(" New correlation: \n Residual sum of squares: %.2f" \
                          % mean((regr.predict(x) - y) ** 2) + '\n')
print('\n Score: %.2f' % regr.score(x, y) + '\n')