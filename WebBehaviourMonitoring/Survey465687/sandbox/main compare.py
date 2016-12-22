from Survey465687 import *
import os
import pandas
from numpy import *
import time
from pygame import mixer

tic = time.clock()

dataset_path = "../data/mousefile_final/"

id = ['.']
#id = ['46.140.117.182']
survey = '465687'
group = 'OldMaximizer' # items 18
items = 18

# group == "NewMaximizer"
# items == 34

# group == "NEO"
# items == 12  # survey 465687
# items == 60  # survey 935959

# group == "ARES"
# items == 20

# group == "SPF"
# items == 16

# group == "HAKEMP"
# items == 27

sample_freq = .5
old_id = 0
old_group = 0
old_data = 0
array_lost_samples = []
error_samples = []
survey_info, survey_scales = survey_results("../data/max_results/Survey results.txt", survey)

df = pandas.DataFrame({'MAX': survey_scales['max'].tolist(),
                           'REGRET': survey_scales['regret'].tolist(),
                           'NEURO': survey_scales['neuro'].tolist(),
                           'NEGAFFECT': survey_scales['neg_affect'].tolist(),
                           'SELFREPR': survey_scales['self_repr'].tolist(),
                           'SATISF': survey_scales['satisf'].tolist(),
                           'DECDIFF': survey_scales['dec_diff'].tolist(),
                           'ALTSEARCH': survey_scales['alt_search'].tolist()},
                          index=survey_info['person_id'])
server_info = pandas.DataFrame()
context_variables = pandas.DataFrame()
asserts = pandas.DataFrame({'nr_complete_survey_subj': [len(df['MAX'])],
                            'nr_files_lost_samples': [0],
                            'nr_split_files': [0],
                            'nr_files': [0],
                            'nr_pointer_files': [0],
                            'nr_no_group': [0],
                            'answer_pos': [0],
                            'final_nr_subjs': [0],
                            'nr_fail_scroll': [0]
                            })
variables_range = pandas.DataFrame({'vt': [],
                                    'vx': [],
                                    'vy': [],
                                    'a': [],
                                    'jerk': [],
                                    't_quest': [],
                                    't_pauses': [],
                                    'l_strokes': [],
                                    's': [],
                                    'straightness': [],
                                    'jitter': [],
                                    'angles': [],
                                    'w': [],
                                    'curvatures': [],
                                    'var_curvatures': [],
                                    'nr_pauses': [],
                                    'nr_scroll': [],
                                    'nr_items_scroll': [],
                                    'nr_correct_within_item': [],
                                    'nr_correct_between_item': [],
                                    'nr_revisit': [],
                                    'inter_item_interval': [],
                                    'time_click': [],
                                    'nr_abandons': []
                                    })

results = pandas.read_csv("../data/table_features/features_OldMaximizer.csv", sep=";", index_col=0)
results = results[isfinite(results['vx_max'])]
temp_x = results[results.columns[8:]]

y = results[results.columns[2]]

save_model = False
use_model = True
if save_model:
    time_array = [Inf]
else:
    time_array = [Inf]

for crop_time in time_array:
    text_file = open("../data/regression/regression_subjotal_featurestotal_usemodel_" + str(crop_time) + "sec.txt", "w")
    for ii in id:
        mse_file = get_files(dataset_path, [ii])
        mse_file = sorted(mse_file)
        for i, mfile in enumerate(mse_file):
            g = open(mfile, 'r')
            if g.readline() != "" and len(g.readlines()) > 2:
                data = pandas.read_csv(mfile, sep='\t', header=None)
                server_info = get_person_id(mfile, dataset_path, server_info)
                server_info = get_survey_id(mfile, server_info)
                server_info = get_step(mfile, server_info)
                if server_info['survey_id'][0] == survey:
                    subj_pos = get_subj_pos(survey_info, server_info)
                    data, lost_samples = reorder_data(data)
                    if lost_samples[0] > 0:
                        asserts['nr_files_lost_samples'] = [asserts['nr_files_lost_samples'][0] + 1]
                    array_lost_samples.append(lost_samples)
                    asserts['lost_samples'] = [mean(array_lost_samples)]
                    track_variables = get_parameters(data)
                    track_variables = extract_item_number(track_variables)
                    i_change_question, items_order = get_new_item_ix(track_variables)
                    context_variables = count_items(items_order, context_variables, survey)
                    if is_new_step(server_info, context_variables, old_id, old_group) == 0:
                        data = pandas.concat([old_data, data])
                        asserts['nr_split_files'] = [asserts['nr_split_files'][0] + 1]
                    track_variables = get_parameters(data)
                    asserts['nr_files'] = [asserts['nr_files'][0] + 1]
                    if not is_tablet(track_variables):
                        asserts['nr_pointer_files'] = [asserts['nr_pointer_files'][0] + 1]
                        _track_variables = track_variables.copy()
                        track_variables = extract_item_number(track_variables)
                        file_error = assert_answer_position(track_variables, group)
                        i_change_question, items_order = get_new_item_ix(track_variables)
                        track_variables['items_order'] = [items_order]
                        old_i_cq = i_change_question
                        old_itemsord = items_order
                        i_change_question, nr_items_orig, t_abandon, track_variables, context_variables = \
                            parameters_analysis(i_change_question, track_variables, context_variables, survey)
                        _t = track_variables['t'][0]
                        context_variables = count_items(items_order, context_variables, survey)
                        if context_variables['group'][0] == 'no group recognized':
                            asserts['nr_no_group'] = [asserts['nr_no_groups'][0] + 1]
                        if context_variables['group'][0] == group and subj_pos != -1:
                            if file_error != 0:
                                asserts['answer_pos'] = [asserts['answer_pos'][0] + 1]
                            else:
                                print(mfile)
                                print(subj_pos)
                                #plot_path(server_info, track_variables, _t)
                                asserts['final_nr_subjs'] = [asserts['final_nr_subjs'][0] + 1]
                                if nr_items_orig != context_variables['nr_items'][0]:
                                    asserts['nr_fail_scroll'] = [asserts['nr_fail_scroll'][0] + 1]
                                track_variables, samples_correction = correct_parameters(track_variables)
                                if samples_correction > 0:
                                    error_samples.append(samples_correction)
                                    asserts['mean_samples_correction'] = [mean(error_samples)]
                                time_variables, space_variables, context_variables = interpolate_data(track_variables,
                                                                                                      context_variables,
                                                                                                      t_abandon, crop_time)
                                df, variables_range = save_results(df, subj_pos, time_variables, space_variables,
                                                                   context_variables, variables_range)
                                #plot_path_frac(subj_pos, space_variables, _track_variables)
                                #multilineplot_zones(subj_pos, _t, time_variables, track_variables, i_change_question, 30)

                    old_id = server_info['person_ip'][0]
                    old_data = data
                    old_group = context_variables['group'][0]
    #violin_results(variables_range)
    #asserts.to_csv("../data/table_features/asserts_" + group + ".csv", sep=";")
    df.to_csv("../data/table_features/featurestest_" + group + ".csv", sep=";")
    df = df[isfinite(df['vx_max'])]
    t = "OLDMAX results"
    for i in [3]:
        t += "Total features results: \n"
        if save_model:
            new_t = regr_total(temp_x, y, t)
            t += new_t
        if use_model:
            features_time = df[df.columns.values[8:]]
            features_time = array(features_time)
            x_time = array([features_time[j, :] for j in arange(0, len(y))])
            loaded_model = cPickle.load(open('model.sav', 'rb'))
            error = loaded_model.predict(x_time)
            outliers = (len(find(error > 5)) + len(find(error < 0))) / len(error)
            t += '\n Score: %.2f' % loaded_model.score(x_time, y)
            t += '\n Residual sum of squares:' + str(mean((loaded_model.predict(x_time) - y) ** 2))
            t += '\n Outliers:' + str(outliers)

    text_file.write(t)
    text_file.close()


toc = time.clock()
t_processing = toc-tic
print(t_processing)
