from WebBehaviourMonitoring.Survey465687 import *
import os
import pandas
from numpy import *
import time
from scipy.signal import decimate
from scipy import interpolate
import matplotlib.pyplot as plt

def train_block(model, signals, signal2model, signal_indexes=None, n_for_each=12, overlap=0.33, random_training=True,
                start_index=0, track_loss=None):
    """
    This method embraces several datasets (or one) according to a number of records for each

    :param signals: - list - a list containing two int vectors:
                                signal[0] - input vector X, used for the input;
                                signal[1] - label vector Y, used for training;

    :param signal2model: - Signal2Model object - object containing the information about the model, for more info
                            check Biosignals.utils.functions.signal2model

    :param signal_indexes: - list - a list containing the indexes of the "signals" variable to be trained.
                                    If None is given, all signals will be used.

    :param n_for_each: - int - number of windows from each signal to be inserted in the model training

    :param overlap: - float - value in the interval [0,1] that corresponds to the overlapping ratio of windows

    :param random_training: - boolean - value that if True random windows will be inserted in the training

    :param start_index: - int - value from which the windows will be selected

    :param track_loss: - boolean - value to plot loss as the model is trained

    :return: trained model
    """

    if signal_indexes is None:
        signal_indexes = range(len(signals[0]))

    model.save(signal2model.signal_directory, model.get_file_tag(-1, -1))

    x_train = []
    y_train = []
    for i in signal_indexes:

        # Creation of the Time Windows from the dataset
        X_windows, y_end_values, n_windows, last_index = segment_signal(signals[0][i], signal2model.window_size,
                                                                        overlap=overlap, start_index=start_index)
        Y_windows, y_end_values, n_windows, last_index = segment_signal(signals[1][i], signal2model.window_size,
                                                                        overlap=overlap, start_index=start_index)
        # List of the windows to be inserted in the dataset
        if random_training:
            window_indexes = np.random.permutation(n_windows)  # randomly select windows
        else:
            window_indexes = list(range((n_windows)))  # first windows are selected

        # Insertion of the windows of this signal in the general dataset
        if len(x_train) == 0:
            # First is for train data
            x_train = X_windows[window_indexes[0:n_for_each], :]
            y_train = Y_windows[window_indexes[0:n_for_each], :]

            # The rest is for test data
            x_test = X_windows[window_indexes[n_for_each:], :]
            y_test = Y_windows[window_indexes[n_for_each:], :]
        else:
            x_train = np.append(x_train, X_windows[window_indexes[0:n_for_each], :], axis=0)
            y_train = np.append(y_train, Y_windows[window_indexes[0:n_for_each], :], axis=0)
            x_test = np.append(x_train, X_windows[window_indexes[n_for_each:], :], axis=0)
            y_test = np.append(x_train, Y_windows[window_indexes[n_for_each:], :], axis=0)

    # Save test data
    model.save_test_data(model.get_file_tag(-1,-1), signal2model.signal_directory, [x_test, y_test])

    # Start time recording
    model.start_time = time.time()
    t1 = time.time()

    # Start training model
    model.train_with_msgd(x_train, y_train, signal2model.number_of_epochs, 0.9, track_loss,
                                 signal2model.signal_directory, 0, save_distance=signal2model.save_interval)

    print("Dataset trained in: ~%d seconds" % int(time.time() - t1))

    # Model last training is then saved
    model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))

# tic = time.clock()
#
# dataset_path = "../data/mousefile_final/"
#
# id = ['.']
# survey = '465687'
# group = 'OldMaximizer'  # items 18
# items = 18
#
# sample_freq = .5
# old_id = 0
# old_group = 0
# old_data = 0
# array_lost_samples = []
# error_samples = []
# survey_info, survey_scales = survey_results("../data/max_results/Survey results.txt", survey)
#
# if os.path.isfile("../data/table_features/features_" + group + ".csv"):
#     df = pandas.read_csv("../data/table_features/features_" + group + ".csv", sep=';', index_col=0)
#     df = df.fillna('')
# else:
#     df = pandas.DataFrame({'MAX': survey_scales['max'].tolist(),
#                            'REGRET': survey_scales['regret'].tolist(),
#                            'NEURO': survey_scales['neuro'].tolist(),
#                            'NEGAFFECT': survey_scales['neg_affect'].tolist(),
#                            'SELFREPR': survey_scales['self_repr'].tolist(),
#                            'SATISF': survey_scales['satisf'].tolist(),
#                            'DECDIFF': survey_scales['dec_diff'].tolist(),
#                            'ALTSEARCH': survey_scales['alt_search'].tolist()},
#                           index=survey_info['person_id'])
#
# server_info = pandas.DataFrame()
# context_variables = pandas.DataFrame()
# asserts = pandas.DataFrame({'nr_complete_survey_subj': [len(df['MAX'])],
#                             'nr_files_lost_samples': [0],
#                             'nr_split_files': [0],
#                             'nr_files': [0],
#                             'nr_pointer_files': [0],
#                             'nr_no_group': [0],
#                             'answer_pos': [0],
#                             'final_nr_subjs': [0],
#                             'nr_fail_scroll': [0]
#                             })
#
# web_monitoring_info = list(range(100))
# for i in list(range(100)):
#     web_monitoring_info[i] = [[],[],[],[]]
#
# person_index =0
# for ii in id:
#     mse_file = get_files(dataset_path, [ii])
#     mse_file = sorted(mse_file)
#     for i, mfile in enumerate(mse_file):
#         g = open(mfile, 'r')
#         if g.readline() != "" and len(g.readlines()) > 2:
#             data = pandas.read_csv(mfile, sep='\t', header=None)
#             server_info = get_step(mfile, get_survey_id(mfile, get_person_id(mfile, dataset_path, server_info)))
#             if server_info['survey_id'][0] == survey:
#                 subj_pos = get_subj_pos(survey_info, server_info)
#                 data, lost_samples = reorder_data(data)
#                 if lost_samples[0] > 0:
#                     asserts['nr_files_lost_samples'] = [asserts['nr_files_lost_samples'][0] + 1]
#                 array_lost_samples.append(lost_samples)
#                 asserts['lost_samples'] = [mean(array_lost_samples)]
#                 track_variables = get_parameters(data)
#                 track_variables = extract_item_number(track_variables)
#                 i_change_question, items_order = get_new_item_ix(track_variables)
#                 context_variables = count_items(items_order, context_variables, survey)
#                 if is_new_step(server_info, context_variables, old_id, old_group) == 0:
#                     data = pandas.concat([old_data, data])
#                 track_variables = get_parameters(data)
#                 if not is_tablet(track_variables):
#                     track_variables = extract_item_number(track_variables)
#                     file_error = assert_answer_position(track_variables, group)
#                     i_change_question, items_order = get_new_item_ix(track_variables)
#                     i_change_question, nr_items_orig, t_abandon, track_variables, context_variables = \
#                         parameters_analysis(i_change_question, track_variables, context_variables, survey)
#                     if context_variables['group'][0] == group and subj_pos != -1:
#                         track_variables, samples_correction = correct_parameters(track_variables)
#                         time_variables, space_variables, context_variables = interpolate_data(track_variables,
#                                                                                                   context_variables,
#                                                                                                   t_abandon)
#
#                         xt = time_variables['xt'].tolist()
#                         yt = time_variables['yt'].tolist()
#                         t = time_variables['tt'].tolist()
#                         max = df.loc[int(subj_pos),'MAX']
#                         web_monitoring_info[person_index][0] = t
#                         web_monitoring_info[person_index][1] = xt
#                         web_monitoring_info[person_index][2] = yt
#                         web_monitoring_info[person_index][3] = max
#                         fs = (1 / diff(web_monitoring_info[person_index][0]))[0,0]
#                         print(fs)
#                         person_index += 1
#
#                 old_id = server_info['person_ip'][0]
#                 old_data = data
#                 old_group = context_variables['group'][0]
#
#
# savez('web_monitoring_info.npz', web_monitoring_info=web_monitoring_info)
# new_web_monitoring_info = web_monitoring_info
# for person_index in range(len(web_monitoring_info)):
#     try:
#         fs = (1 / diff(web_monitoring_info[person_index][0]))[0, 0]
#         factor = 250 / fs
#         x = 0
#         y = 0
#         new_t = np.arange(0, len(t) * 250 / fs, 1 / 250)
#         if fs > 250:
#             x = decimate(web_monitoring_info[person_index][1])
#             y = decimate(web_monitoring_info[person_index][2])
#         elif fs < 250:
#             tck_x = interpolate.splrep(t, web_monitoring_info[person_index][1], s=0)
#             tck_y = interpolate.splrep(t, web_monitoring_info[person_index][2], s=0)
#             x = interpolate.splev(new_t, tck_x, der=0)
#             y = interpolate.splev(new_t, tck_y, der=0)
#
#         new_web_monitoring_info[person_index][0] = np.asarray(new_t)[0]
#         new_web_monitoring_info[person_index][1] = np.asarray(x)[0]
#         new_web_monitoring_info[person_index][2] = np.asarray(y)[0]
#     except:
#         pass

# savez('web_monitoring_info.npz',
#       web_monitoring_info=web_monitoring_info,
#       new_web_monitoring_info=new_web_monitoring_info)

npzfile = load('web_monitoring_info.npz')
web_monitoring_info, new_web_monitoring_info = npzfile["web_monitoring_info"], npzfile["new_web_monitoring_info"]

import BioSignalsDeepLibphys.models.libphys_MBGRU as GRU
import BioSignalsDeepLibphys.utils.functions.database as db
from BioSignalsDeepLibphys.utils.functions.common import segment_signal, process_web_signal, process_signal
from BioSignalsDeepLibphys.utils.functions.signal2model import Signal2Model


signal_dim = 64
hidden_dim = 256
batch_size = 64
n_for_each = 32
W = 256


signal_directory = 'WEB_[{0}.{1}]'.format(signal_dim, batch_size)
max_list = asarray([new_web_monitoring_info[person_index][3] for person_index in range(93)])
# print(len(where(max_list < 2.25)[0]))
# print(len(where(max_list > 3.25)[0]))
# print(len(where(logical_and(max_list <= 3.5,max_list >= 2.25))[0]))


max_groups = [  where(logical_and(max_list < 2.5,max_list >= 1))[0],
                where(logical_and(max_list < 3.5,max_list >= 3.0))[0],
                where(logical_and(max_list < 3.0, max_list >= 2.5))[0],
                where(logical_and(max_list < 5, max_list >= 3.5))[0]]


signal_name = "web_monitoring"
for person_index in range(24, len(new_web_monitoring_info)):
    try:
        t = new_web_monitoring_info[person_index][0][0]
        x = new_web_monitoring_info[person_index][1][0]
        y = new_web_monitoring_info[person_index][2][0]
        max = new_web_monitoring_info[person_index][3]

        sigx = signal_name + "_x[" + str(person_index) + "." + str(max)+"]"
        sigy = signal_name + "_y[" + str(person_index) + "." + str(max)+"]"
    # Load Model

        x = process_signal(x, signal_dim, 10, False, None, False)
        y = process_signal(y, signal_dim, 10, False, None, False)

        signal_x = Signal2Model(sigx, signal_directory, signal_dim=signal_dim, mini_batch_size=16)
        signal_y = Signal2Model(sigy, signal_directory, signal_dim=signal_dim, mini_batch_size=16)
        if len(x) < (signal_x.batch_size*signal_x.window_size/3):
            print("Person {0} was not processed due to lack of data".format(person_index))
            pass
        else:
            if person_index != 14:
                model = GRU.LibPhys_GRU(signal_dim=64, signal_name=sigx, hidden_dim=hidden_dim, n_windows=16)
                model.save(signal_directory, model.get_file_tag(-1, -1))

                model.train_signal(x[:-1], x[1:], signal_x, decay=0.95, track_loss=False, save_distance=100000)

            model = GRU.LibPhys_GRU(signal_dim=64, signal_name=sigy, hidden_dim=hidden_dim, n_windows=16)
            model.save(signal_directory, model.get_file_tag(-1, -1))

            model.train_signal(x[:-1], x[1:], signal_y, decay=0.95, track_loss=False, save_distance=100000)
    except:
        print("Exception raised on person {0}".format(person_index))
        pass

# signal_name = "web_group"
# training_indexes = [[],[],[],[]]
# testing_indexes = [[],[],[],[]]
# for i in range(0, len(max_groups)):
#     group = max_groups[i]
#     random_indexes = np.random.permutation(len(group))
#     training_indexes[i] = random_indexes[:int(len(group))]
#     testing_indexes[i] = []#random_indexes[int(len(group)*0.7):]
#     x = [[], []]
#     y = [[], []]
#
#     for person_index in group:
#         x_temp = process_web_signal(new_web_monitoring_info[person_index][1][0], signal_dim, 10, False)
#         y_temp = process_web_signal(new_web_monitoring_info[person_index][2][0], signal_dim, 10, False)
#         # plt.plot(x_temp)
#         # plt.plot(y_temp)
#         # plt.show()
#         x[0].append(x_temp[: -1])
#         x[1].append(x_temp[1:])
#         y[0].append(y_temp[: -1])
#         y[1].append(y_temp[1:])
#
#     max = mean(max_list[group])
#
#     sigx = signal_name + "_x[" + str(i) + "." + str(max)+"]"
#     sigy = signal_name + "_y[" + str(i) + "." + str(max)+"]"
#     # Load Model
#
#     signal_x2model = Signal2Model(signal_directory=signal_directory, signal_dim=signal_dim, model_name=sigx,
#                                   hidden_dim=hidden_dim, window_size=W)
#
#     signal_y2model = Signal2Model(signal_directory=signal_directory, signal_dim=signal_dim, model_name=sigy,
#                                   hidden_dim=hidden_dim, window_size=W)
#
#     model = GRU.LibPhys_GRU(signal_dim=signal_x2model.signal_dim, signal_name=sigx, hidden_dim=hidden_dim, n_windows=signal_x2model.mini_batch_size)
#     train_block(model, x, signal_x2model, n_for_each=n_for_each)
#
#     model = GRU.LibPhys_GRU(signal_dim=signal_y2model.signal_dim, signal_name=sigy, hidden_dim=hidden_dim, n_windows=signal_y2model.mini_batch_size)
#     train_block(model, y, signal_y2model, n_for_each=n_for_each)
#
#     savez('web_monitoring_info_trained_128.npz',
#           web_monitoring_info=web_monitoring_info,
#           new_web_monitoring_info=new_web_monitoring_info,
#           max_groups=max_groups,
#           group=group,
#           training_indexes = training_indexes,
#           testing_indexes = training_indexes)

# toc = time.clock()
# t_processing = toc-tic
# print(t_processing)


