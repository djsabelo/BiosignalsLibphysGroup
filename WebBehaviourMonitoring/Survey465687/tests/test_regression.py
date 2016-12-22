import pandas
from numpy import *
from scipy import *
from pylab import plt
from sklearn import linear_model
from sklearn.cross_validation import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D
from regression_tools import *

results = pandas.read_csv(".\\data\\table16subj.csv", sep=";")
y = results[results.columns[2]]
temp_x = normalize(results[results.columns[4:]])

print("Total features results: \n")
regr_total(results, y)

print "\n Greedy Forward Selection total features results: \n"
regr_iteration(temp_x, y)

print "\n Greedy Forward Selection LeaveOneOut results: \n"
features = regr_leavoneout(temp_x, y)

print "\n Greedy Forward Selection Results Score (Test on train): \n"
regr_score(features, results, y)
'''
# Plot outputs
plt.scatter(array(temp_x['vMin']), y, color='blue')
plt.scatter(array(temp_x['nr_correct_within_item']), y,  color='blue')
plt.scatter(array(temp_x['nr_pauses']), y,  color='blue')
plt.scatter(array(temp_x['vMovingMin']), y,  color='blue')
plt.scatter(array(temp_x['straightnessMax']), y,  color='blue')
plt.scatter(array(temp_x['nrZapp']), y,  color='blue')
plt.plot(new_x, regr.predict(new_x), '.y')
plt.show()

temp_x = normalize(results[['vMin', 'nr_correct_within_item']])
new_x = [[] for i in arange(0, len(y))]
for jj in temp_x.columns:
    for ix in arange(0, len(y)):
        new_x[ix].append(temp_x[jj][ix])
regr.fit(new_x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = arange(-1, 3.5, 1)
Y = arange(-1, 5, 1)
x_surf, y_surf = meshgrid(X, Y)
z = regr.predict(new_x)
for i in arange(0, len(array(temp_x['vMin']))):
    ax.scatter(array(temp_x['vMin'])[i], array(temp_x['nr_correct_within_item'])[i], y[i],  zdir='z')
print z.shape
print x_surf.shape
print z.reshape(2,8)
ax.plot_surface(x_surf, y_surf, z.reshape(x_surf.shape), rstride=1, cstride=1, color='None')
plt.show()
'''