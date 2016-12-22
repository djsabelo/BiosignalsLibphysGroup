from Survey465687.sandbox.regression_tools import *
import pandas
from pylab import*
import random
import numpy as np
from sklearn import linear_model, model_selection, metrics


results = pandas.read_csv("../data/table_features/features_OldMaximizer_[41, 109, 134, 69].csv", sep=",", index_col=0)
results = results[isfinite(results['vx_max'])]

Features_x = ['s', 'inter_item_interval_min', 'nr_correct_within_item', 'jitter', 'w_max', 'inter_item_interval_mean', 'l_strokes_min', 'curvatures_max', 'inter_item_interval_max', 'a_mean', 't_pauses_min', 'vy_min', 'inter_item_interval_std', 't_pauses_max', 't_quest_mean', 'l_strokes_std', 'vy_mean', 'vy_std', 'straightness_min', 'straightness_max', 'time_click_max', 'curvatures_mean']
Features_x = ['s', 'inter_item_interval_min', 'nr_correct_within_item', 'jitter', 'w_max', 'inter_item_interval_mean', 'l_strokes_min', 'curvatures_max', 'inter_item_interval_max', 'a_mean', 't_pauses_min', 'vy_min', 'inter_item_interval_std', 't_pauses_max', 't_quest_mean', 'l_strokes_std', 'vy_mean', 'vy_std', 'straightness_min', 'straightness_max', 'time_click_max', 'curvatures_mean', 'l_strokes_max', 't_pauses_std', 't_quest_std', 'nr_items_scroll', 'straightness_std', 'straightness_mean', 'time_click_mean', 'angles_std', 'vt_min', 'var_curvatures_mean', 'nr_revisit', 'time_click_min', 'w_std', 'nr_pauses', 'jerk_mean', 't_pauses_mean', 'time_click_std', 'vy_max', 'vx_max', 'a_max', 't_quest_min', 'a_min', 'vt_max', 'nr_scroll', 'jerk_max', 'vx_min', 'angles_mean', 'angles_max', 'w_min', 'jerk_min', 'jerk_std', 'w_mean', 'angles_min', 'a_std', 'vt_std', 'vx_std', 'vx_mean', 'var_curvatures_min', 'vt_mean', 'l_strokes_mean', 'nr_abandons', 'curvatures_min', 'curvatures_std', 'var_curvatures_max', 'nr_correct_between_item', 'var_curvatures_std']
#Features_x = ['s', 'inter_item_interval_min', 'jitter', 'inter_item_interval_mean', 'vx_std', 'vy_min', 'vx_min', 'l_strokes_min', 'l_strokes_std', 't_quest_mean', 'nr_correct_within_item']
Features = results.columns.tolist()
normalization_factors = pandas.DataFrame(index=Features_x)


arr = arange(84)
np.random.shuffle(arr)
_results = results.ix[results.index.get_values()[arr[:43]]]
for j in Features[8:]:
    normalization_factors.loc[j, 'Mean'] = mean(_results[j].tolist())
    normalization_factors.loc[j, 'STD'] = std(_results[j].tolist())
    _results[j] = normalize_ms(_results[j].tolist(), mean(_results[j].tolist()), std(_results[j].tolist()))

results_ = results.ix[results.index.get_values()[arr[43:]]]
for j in Features[8:]:
    results_[j] = normalize_ms(results_[j].tolist(), normalization_factors.loc[j, 'Mean'], normalization_factors.loc[j, 'STD'])

y_true_train = _results[_results.columns[2]]
y_true_test = results_[results_.columns[2]]


_results = _results[Features]
results_ = results_[Features]

x_true_train = _results[_results.columns[:]]
x_true_test = results_[results_.columns[:]]

x_fake = results[results.columns[:]][Features]
for j in x_fake.columns.tolist():
    x = x_fake[j].tolist()
    for i in arange(0, len(x_fake[j])):
        x_fake.loc[x_fake.index.values[i], j] = random.choice(x)

x_fake_train = x_fake[:43]
x_fake_test = x_fake[43:]

x_true_train = array([array(x_true_train)[i, :] for i in arange(0, len(x_true_train))])
x_true_test = array([array(x_true_test)[i, :] for i in arange(0, len(x_true_test))])
x_fake_train = array([array(x_fake_train)[i, :] for i in arange(0, len(x_fake_train))])
x_fake_test = array([array(x_fake_test)[i, :] for i in arange(0, len(x_fake_test))])
regr = linear_model.LinearRegression()
regr.fit(x_true_train, y_true_train)
print(metrics.r2_score(y_true_test, regr.predict(x_true_test)))
print(metrics.r2_score(y_true_test, regr.predict(x_fake_test)))
regr = linear_model.LinearRegression()
regr.fit(x_fake_train, y_true_train)
print(metrics.r2_score(y_true_test, regr.predict(x_fake_test)))
print(metrics.r2_score(y_true_test, regr.predict(x_true_test)))