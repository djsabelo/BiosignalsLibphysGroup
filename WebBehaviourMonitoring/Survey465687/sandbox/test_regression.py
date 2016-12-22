from Survey465687.sandbox.regression_tools import *
from sklearn import preprocessing
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

    fxn()
results = pandas.read_csv("../data/table_features/features_OldMaximizer_[41, 109, 134, 69].csv", sep=",", index_col=0)
results = results[isfinite(results['vx_max'])]

# results_time = pandas.read_csv("../data/table_features/features_OldMaximizer_30sec.csv", sep=",")
# results_time = results_time[isfinite(results_time['vx_max'])]

subj = "total"
feat = "total"
if subj == "extremes":
    results = pandas.concat([results.loc[lambda df: df.MAX > 3.75, :], results.loc[lambda df: df.MAX < 2.25, :]])
if subj == "middle":
    results = pandas.concat([results.loc[lambda df: df.MAX < 3.75, :], results.loc[lambda df: df.MAX > 2.25, :]])

if feat == "non-correlated":
    results = results.drop(['jerk_max', 'jerk_std', 'w_std', 'curvatures_std', 'curvatures_min', 'var_curvatures_mean',
                            'var_curvatures_std', 'inter_item_interval_std', 'time_click_std'], axis=1)

features = ['s', 'inter_item_interval_min', 'jitter', 'inter_item_interval_mean', 'vx_std', 'vy_min', 'vx_min', 'l_strokes_min', 'l_strokes_std', 't_quest_mean', 'nr_correct_within_item']
temp_x = results[results.columns[9:]]
temp_x = temp_x[features]
#for j in temp_x.columns.tolist():
#    temp_x[j] = preprocessing.normalize(temp_x[j].tolist())[0]
#temp_x = temp_x[['inter_item_interval_min', 's', 'nr_correct_within_item', 'vx_min', 'jitter', 'inter_item_interval_mean', 'vx_std', 'a_mean', 'vy_min', 'l_strokes_min', 't_quest_mean', 'l_strokes_std', 'straightness_max', 'vt_min', 'w_std']]

text_file = open("../data/regression/regression_subj" + subj + "_features" + feat + "loo_[41, 109, 134, 69]_11features.txt", "w")


t = "OLDMAX results \n"
for i in [2]:
    y = results[results.columns[i]]
    '''

    t += "Total features results: \n"
    new_t = regr_total(temp_x, y, t)

    new_t = t
    new_t += "\n Greedy Forward Selection total features results: \n"
    new_t = regr_iteration(temp_x, y, new_t)
    '''
    new_t = t
    new_t += "\n Greedy Forward Selection LeaveOneOut results: \n"
    features, new_t = test_regr_leavoneout(temp_x, y, new_t)
    '''
    new_t += "\n Greedy Forward Selection Results Score (Test on train): \n"
    new_t = regr_score(features, results, y, new_t)
'''
text_file.write(new_t)

text_file.close()

