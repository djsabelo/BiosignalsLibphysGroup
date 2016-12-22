from scipy import *
import pandas
from sklearn import linear_model


def regression_total(res, y, ind, t, t2):
    y = y.reshape(len(y), 1)

    for i in res.columns.values[ind:]:
        features_name = i
        features = res[i]
        features = array(features)
        x = features
        x = x.reshape(len(x), 1)
        r = linear_model.LinearRegression()
        r.fit(x, y)

        t += str(features_name) + 'Score: %.2f' % r.score(x, y) + '\n'
        if r.score(x, y) > 0.95:
            t += "ALERT \n"
            t2 += str(features_name) + 'Score: %.2f' % r.score(x, y) + '\n'
    t += '\n'
    t2 += '\n'

    return t, t2

results = pandas.read_csv("../data/table_features/features_OldMaximizer.csv", sep=";")
results = results[isfinite(results['vx_max'])]
text = "OLDMAX features correlation \n"
text2 = text

text_file = open("../data/regression/featuresCorrelation.txt", "w")
text2_file = open("../data/regression/featuresCorrelation95.txt", "w")

for j in arange(9, 78):
    scale = results[results.columns[j]]
    text += str(results.columns.values[j]) + '\n' + "Total features results: \n"
    text2 += str(results.columns.values[j]) + '\n' + "Total features results: \n"
    text, text2 = regression_total(results, scale, 9, text, text2)
text_file.write(text)
text2_file.write(text2)


text_file.close()
text2_file.close()
