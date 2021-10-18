import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from utils import read_data, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score
plt.ion()
plt.rcParams.update({'font.size': 22})

classes = np.array(['NOCS','SOCS','RICS','TRICS'])


def read_Xy():

    X_csv = 'data/data_traces_so/DES009_Night_4_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES009_Night_4_Class_Y.csv'
    X9_1, y9_1, features, scaler9_1 = read_data(X_csv, y_csv,
                                                drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)

    X_csv = 'data/data_traces_so/DES009_Night_5_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES009_Night_5_Class_Y.csv'
    X9_2, y9_2, features, scaler9_2 = read_data(X_csv, y_csv,
                                                drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)


    X_csv = 'data/data_traces_so/DES004_Night_8_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES004_Night_8_Class_Y.csv'
    X4_1, y4_1, features, scaler4_1 = read_data(X_csv, y_csv,
                                                drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)

    X_csv = 'data/data_traces_so/DES004_Night_9_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES004_Night_9_Class_Y.csv'
    X4_2, y4_2, features, scaler4_1 = read_data(X_csv, y_csv,
                                                drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)

    X_csv = 'data/data_traces_so/DES012_POD1_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES012_POD1_Class_Y.csv'
    X12_1, y12_1, features, scaler12_1 = read_data(X_csv, y_csv,
                                                   drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)

    X_csv = 'data/data_traces_so/DES012_POD2_SpindleFeatures_X.csv'
    y_csv = 'data/data_traces_so/DES012_POD2_Class_Y.csv'
    X12_2, y12_2, features, scaler12_1 = read_data(X_csv, y_csv,
                                                   drop_columns=['startSample', 'endSample'], scale=True, so_nan_avoid=True)

    X = np.concatenate((X9_1, X9_2, X4_1, X4_2, X12_1, X12_2), axis=0)
    y = np.concatenate((y9_1, y9_2, y4_1, y4_2, y12_1, y12_2))

    nocs_rics = np.isin(y, [1, 3])
    socs_trics = np.isin(y, [2, 4])

    X_nocs_rics = X[nocs_rics, :14]
    y_nocs_rics = y[nocs_rics]

    X_socs_trics = X[socs_trics, :]
    y_socs_trics = y[socs_trics]

    not_nan_ind = ~np.isnan(X_nocs_rics.mean(1))
    X_nocs_rics = X_nocs_rics[not_nan_ind, :]
    y_nocs_rics = y_nocs_rics[not_nan_ind]

    not_nan_ind = ~np.isnan(X_socs_trics.mean(1))
    X_socs_trics = X_socs_trics[not_nan_ind, :]
    y_socs_trics = y_socs_trics[not_nan_ind]

    X_socs_trics, y_socs_trics = shuffle(X_socs_trics, y_socs_trics)
    X_nocs_rics, y_nocs_rics = shuffle(X_nocs_rics, y_nocs_rics)

    return X_socs_trics, y_socs_trics, X_nocs_rics, y_nocs_rics


def run_classfier(Xn, yn, selection, test_size=0.25, minority_fraction=0.95):
    X_train, X_test, y_train, y_test = train_test_split(Xn, yn, stratify=yn, test_size=test_size)
    minority_class = selection[np.array([(y_train == item).sum() for item in selection]).argmin()]
    sampling_strategy = {item: round(minority_fraction*(y_train == minority_class).sum()) for item in selection}
    clf = BalancedRandomForestClassifier(n_estimators=1000,
                                         sampling_strategy=sampling_strategy,
                                         class_weight='balanced_subsample', n_jobs=12)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return geometric_mean_score(y_test, y_pred), y_test, y_pred, clf


def plot(results, selection):
    fig, ax = plt.subplots(figsize=(20, 20))
    cm_brf = confusion_matrix(results[1], results[2])
    plot_confusion_matrix(cm_brf, classes=classes[np.array(selection)-1],
                          ax=ax, title=results[3].__repr__() + ' - ' + str(classes[np.array(selection)-1]))
    print('Sampler: {} - Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
          .format(results[3], balanced_accuracy_score(results[1], results[2]),
                  geometric_mean_score(results[1], results[2])))

X_socs_trics, y_socs_trics, X_nocs_rics, y_nocs_rics = read_Xy()
result_socs_trics = run_classfier(X_socs_trics, y_socs_trics, selection = [2, 4])
result_nocs_rics = run_classfier(X_nocs_rics, y_nocs_rics, selection = [1, 3])

plot(result_socs_trics, [2, 4])
plot(result_nocs_rics, [1, 3])