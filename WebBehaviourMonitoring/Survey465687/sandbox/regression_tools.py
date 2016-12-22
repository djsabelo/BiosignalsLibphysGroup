import pandas
from pylab import *
from scipy import *
import _pickle as cPickle
from sklearn import linear_model, model_selection, metrics
#cross_validation
import itertools


def plot_confusion_matrix(cm, classes, frame,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, cmap=cmap, interpolation='none')

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0.5, len(classes) + 0.5)
    tick_marks2 = np.arange(0.5, len(frame) + 0.5)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks2, frame)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # plt.text(j, i, cm[i, j],
    # horizontalalignment="center",
    # color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('n features label')
    plt.xlabel('maximizer label')


def normalize(function):
    return (function[:]-mean(function))/std(function)


def normalize_ms(function, m, s):
    return (function-m)/s


def regr_total_compare(results, results_time, y, ind, tfile):
    y = y.reshape(len(y), 1)

    results_time = results_time[isfinite(results_time['vx_max'])]
    features_time = results_time[results_time.columns.values[ind:]]
    features_time = array(features_time)
    features_name = results.columns.values[ind+1:]

    features = results[results.columns.values[ind+1:]]
    features = array(features)

    x = array([features[i, :] for i in arange(0, len(y))])

    x_time = array([features_time[i, :] for i in arange(0, len(y))])

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    tfile += 'Features:' + str(features_name) + '\n Coefficients:' + str(regr.coef_) + "\n Residual sum of squares: %.2f" \
                                                                                      % mean((regr.predict(x) - y) ** 2)
    tfile += '\n Score: %.2f' % regr.score(x, y)
    error = regr.predict(x_time)
    outliers = (len(find(error > 5)) + len(find(error < 0)))/len(error)
    #y = y[find(error<5)]
    #x_time = x_time[find(error<5)]
    error = error[find(error<5)]

    #y = y[find(error>0)]
    #x_time = x_time[find(error > 0)]
    error = error[find(error>0)]


    tfile += '\n Score: %.2f' % regr.score(x_time, y)
    tfile += '\n Residual sum of squares:' + str(mean((regr.predict(x_time) - y) ** 2))
    tfile += '\n Outliers:' + str(outliers)
    # plt.scatter(y, regr.predict(x_time), color='blue')
    # savefig('../data/regression/Total features regression.pdf')
    # close()

    return tfile, outliers, regr.score(x_time, y), mean((regr.predict(x_time) - y) ** 2)


def regr_total(results, y, tfile):
    y = y.values.reshape(len(y))

    features_name = results.columns.values
    # features = results[results.columns.values[ind:]]
    features = array(results)
    x = array([features[i, :] for i in arange(0, len(y))])
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    tfile += 'Features:' + str(features_name) + '\n Coefficients:' + str(regr.coef_) + "\n Residual sum of squares: %.2f" \
                                                                                      % mean((regr.predict(x) - y) ** 2)+ '\n'
    tfile += '\n Score: %.2f' % regr.score(x, y) + '\n'
    filename = 'model.sav'
    cPickle.dump(regr, open(filename, 'wb'))
    plt.scatter(y, regr.predict(x), color='blue')
    # savefig('../data/regression/Total features regression.pdf')
    # close()
    return tfile


def regr_iteration(temp_x, y, tfile):

    regr = linear_model.LinearRegression()
    scores = []
    msee = []
    all_mse = []
    features = []
    x = []
    while len(temp_x.columns) > 0:  # Force to analyse every feature
        score = []
        a_mse = []
        for i, feature in enumerate(temp_x.columns):   # Compare best features with rest of features
            if len(x) > 0:
                _x = pandas.concat([x, temp_x[feature]], axis=1)
            else:
                _x = pandas.DataFrame(temp_x[feature])
            _x = array(_x)
            new_x = array([_x[j, :] for j in arange(0, len(y))])
            regr.fit(new_x, y)

            #score.append(regr.score(new_x, y))
            score.append(metrics.r2_score(y, regr.predict(new_x)))
            a_mse.append(((regr.predict(new_x) - y) ** 2).tolist())

        scores += [max(score)]
        msee += [a_mse[argmax(score)]]

        features += [temp_x.columns[argmax(score)]]
        nf = temp_x[features[-1]]

        if len(x) > 0:
            x = pandas.concat([x, nf], axis=1)
        else:
            x = pandas.DataFrame(nf)

        temp_x = temp_x.drop(features[-1], 1)
    all_mse = amax(msee, axis=1)/max(amax(msee, axis=1))
    figure(1)
    plot(all_mse, 'o')
    plot(amin(msee, axis=1)/max(amin(msee, axis=1)), '.r')
    plot(mean(msee, axis=1)/max(mean(msee, axis=1)), '.k')
    figure(2)
    m = matrix(msee)
    plot_confusion_matrix(m, classes=temp_x.index.tolist(), frame=arange(1, 70))
    plt.show()
    tfile += 'Features:' + str(features) + '\n'
    tfile += 'Score:' + str(scores) + '\n'
    tfile += 'MSE:' + str(msee) + '\n'

    return tfile


def regr_leavoneout(temp_x, y, tfile):
    regr = linear_model.LinearRegression()
    mse = []
    m_mse = []
    features = []
    x = []
    xtestr = []
    ytestr = []
    msee = []

    while len(temp_x.columns) > 0:  # Force to analyse every feature
        print('o')
        lpo = model_selection.LeaveOneOut()
        y = array(y)
        feat_mse = []
        all_loo_mse = []
        for i, feature in enumerate(temp_x.columns):   # Compare best features with rest of features

            if len(x) > 0:
                _x = pandas.concat([x, temp_x[feature]], axis=1)
            else:
                _x = pandas.DataFrame(temp_x[feature])
            _x = array(_x)
            new_x = [[_x[j, :]] for j in arange(0, len(y))]
            lpo_mse = []
            for train, test in lpo.split(_x):
                xtrain = []
                ytrain = []
                for itest in test:
                    xtest = _x[itest]
                    ytest = y[itest]
                for itrain in train:
                    xtrain.append(_x[itrain])
                    ytrain.append(y[itrain])
                ytrain = array(ytrain)
                reshape(ytrain, len(ytrain))
                regr.fit(xtrain, ytrain)
                xtestr.append(regr.predict(xtest))
                ytestr.append(ytest)
                lpo_mse.append(((regr.predict(xtest) - ytest) ** 2)[0])
                all_loo_mse.append(((regr.predict(xtest) - ytest) ** 2).tolist())
            feat_mse.append(mean(lpo_mse))


        mse += [min(feat_mse)]
        m_mse += [mean(feat_mse)]
        msee += [lpo_mse]
        features += [temp_x.columns[argmin(feat_mse)]]
        nf = temp_x[features[-1]]

        if len(x) > 0:
            x = pandas.concat([x, nf], axis=1)
        else:
            x = pandas.DataFrame(nf)
        temp_x = temp_x.drop(features[-1], 1)

    all_mse = amax(msee, axis=1) / max(amax(msee, axis=1))
    figure(1)
    plot(all_mse, 'o')
    plot(amin(msee, axis=1) / max(amin(msee, axis=1)), '.r')
    plot(mean(msee, axis=1) / max(mean(msee, axis=1)), '.k')
    figure(2)
    m = matrix(msee)
    plot_confusion_matrix(m, classes=temp_x.index.tolist(), frame=arange(1, 16))
    plt.show()

    tfile += 'Features:' + str(features) +'\n'
    tfile += 'MSE:' + str(m_mse) + '\n'

    return features, tfile


def regr_score(features, results, y, tfile):
    score = []
    mse = []
    for i in arange(0, len(features)):
        temp_x = results[features[:i+1]]
        regr = linear_model.LinearRegression()
        temp_x = array(temp_x)
        new_x = array([temp_x[j, :] for j in arange(0, len(y))])
        regr.fit(new_x, y)
        score.append(regr.score(new_x, y))
        mse.append(mean((regr.predict(new_x) - array(y))**2))

    tfile += 'Features:' + str(features) + '\n'
    tfile += 'Score:' + str(score) + '\n'
    tfile += 'MSE:' + str(mse) + '\n'
    # plt.scatter(y, regr.predict(new_x), color='blue')
    # savefig('../data/regression/LOO order test on train regression.pdf')
    # close()
    return tfile

def test_regr_leavoneout(temp_x, y, tfile):
    regr = linear_model.LinearRegression()
    lpo = model_selection.LeaveOneOut()
    mse = []
    m_mse = []
    features = []
    x = []
    msee = []

    xtest = []
    A = []
    _x = pandas.DataFrame()
    for i in temp_x.columns.tolist():
        A.extend([str(i)] * len(y))
    B = array([str(i) for i in arange(0, len(y))] * len(temp_x.columns.tolist()))
    #df = pandas.DataFrame(columns=pandas.MultiIndex.from_tuples(list(zip(A, B))), index=['test', 'train'])
    df = pandas.DataFrame(columns=['test', 'train'], index=pandas.MultiIndex.from_tuples(list(zip(A, B))))
    for train, test in lpo.split(y):
        for f in temp_x.columns.tolist():
            h = [temp_x[f].tolist()[i] for i in train]
            df.loc[(f, str(test[0])), 'train'] = normalize(h)
            df.loc[(f, str(test[0])), 'test'] = normalize_ms(temp_x[f].tolist()[test[0]], m=mean(h), s=std(h))
    while len(temp_x.columns) > 0:  # Force to analyse every feature
        print('o')

        y = array(y)
        feat_mse = []
        all_loo_mse = []
        normalization_factors = pandas.DataFrame(index=temp_x.columns.tolist())

        for i, feature in enumerate(temp_x.columns):   # Compare best features with rest of features

            if len(x) > 0:
                _x = pandas.concat([x, temp_x[feature]], axis=1)
            else:
                _x = pandas.DataFrame(temp_x[feature])
            lpo_mse = []
            for train, test in lpo.split(_x):
                xtrain = []
                xtest = []
                for i, feature_x in enumerate(_x.columns):
                    xtrain.append(df.loc[(feature_x, str(test[0])), 'train'].tolist())
                    xtest.append(df.loc[(feature_x, str(test[0])), 'test'].tolist())
                ytest = y[test]
                ytrain = y[train]
                xtrain = array(xtrain)
                xtrain = xtrain.T
                ytrain = array(ytrain)
                regr.fit(xtrain, ytrain)
                if len(xtest) == 1:
                    xtest = array(xtest).reshape(-1, 1)
                else:
                    pass
                    xtest = [xtest]
                lpo_mse.append(((regr.predict(xtest) - ytest) ** 2)[0])
                all_loo_mse.append(((regr.predict(xtest) - ytest) ** 2).tolist())
            feat_mse.append(mean(lpo_mse))
        mse += [min(feat_mse)]
        m_mse += [mean(feat_mse)]
        msee += [lpo_mse]
        features += [temp_x.columns[argmin(feat_mse)]]
        nf = temp_x[features[-1]]

        if len(x) > 0:
            x = pandas.concat([x, nf], axis=1)
        else:
            x = pandas.DataFrame(nf)
        temp_x = temp_x.drop(features[-1], 1)

    all_mse = amax(msee, axis=1) / max(amax(msee, axis=1))
    figure(1)
    plot(all_mse, 'o')
    plot(amin(msee, axis=1) / max(amin(msee, axis=1)), '.r')
    plot(mean(msee, axis=1) / max(mean(msee, axis=1)), '.k')
    figure(2)
    m = matrix(msee)
    plot_confusion_matrix(m, classes=temp_x.index.tolist(), frame=arange(1, len(features)))
    plt.show()

    tfile += 'Features:' + str(features) + '\n'
    tfile += 'MSE:' + str(m_mse) + '\n'

    return features, tfile

