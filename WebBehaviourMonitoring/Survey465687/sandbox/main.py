from WebBehaviourMonitoring.Survey465687 import *
import os
import pandas
from numpy import *
import time

tic = time.clock()

dataset_path = "../data/mousefile_final/"

id = ['.']
survey = '465687'
group = 'OldMaximizer'  # items 18
items = 18

sample_freq = .5
old_id = 0
old_group = 0
old_data = 0
array_lost_samples = []
error_samples = []
survey_info, survey_scales = survey_results("../data/max_results/Survey results.txt", survey)

if os.path.isfile("../data/table_features/features_" + group + ".csv"):
    df = pandas.read_csv("../data/table_features/features_" + group + ".csv", sep=';', index_col=0)
    df = df.fillna('')
else:
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
                track_variables = get_parameters(data)
                if not is_tablet(track_variables):
                    track_variables = extract_item_number(track_variables)
                    file_error = assert_answer_position(track_variables, group)
                    i_change_question, items_order = get_new_item_ix(track_variables)
                    i_change_question, nr_items_orig, t_abandon, track_variables, context_variables = \
                        parameters_analysis(i_change_question, track_variables, context_variables, survey)
                    if context_variables['group'][0] == group and subj_pos != -1:
                        track_variables, samples_correction = correct_parameters(track_variables)
                        time_variables, space_variables, context_variables = interpolate_data(track_variables,
                                                                                                  context_variables,
                                                                                                  t_abandon)


                old_id = server_info['person_ip'][0]
                old_data = data
                old_group = context_variables['group'][0]


print(track_variables)
print(time_variables)
print(survey_scales)
toc = time.clock()
t_processing = toc-tic
print(t_processing)
