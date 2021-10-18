import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle
from numpy import interp
import itertools
from sklearn.ensemble import ExtraTreesClassifier
from pca import pca
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
plt.ion()
plt.rcParams.update({'font.size': 22})

def autolabel(rects, ax, format_str='{:.3f}'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def pca_diagnostics(X, n_components=2, n_feat=2):
    model = pca(n_components=n_components)
    results = model.fit_transform(X)
    fig, ax = model.plot()
    fig, ax = model.scatter()
    fig, ax = model.biplot(n_feat=n_feat)


def read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample', 'channel'], scale=True, so_nan_avoid=False):
    data = pd.read_csv(X_csv)
    data.drop(drop_columns, axis=1, inplace=True)
    features = np.array(data.keys())

    labels = pd.read_csv(y_csv)
    y = labels.to_numpy().ravel()
    (y == 4).sum()
    X = data.to_numpy()

    if so_nan_avoid:
        not_nan_ind = ~np.isnan(X[:, :14].mean(1))
    else:
        not_nan_ind = ~np.isnan(X.mean(1))

    X = X[not_nan_ind, :]
    y = y[not_nan_ind]
    if scale:
        scaler = MinMaxScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler=None
    return X, y, features, scaler

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')




def plot_feature_corr(X, features, stem='train', save=False):
    df = pd.DataFrame(X)
    fig, ax = plt.subplots(figsize=(25, 25))
    corr = df.corr()
    corr[corr==1] = 0
    corr[(corr <= 0.5) & (corr >= -0.5)] = 0
    hmap = sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, cmap=sns.diverging_palette(0, 230, 90, 60, as_cmap=True),
                annot=True, xticklabels=features, yticklabels=features)
    hmap.set_xticklabels(features, rotation=30)
    hmap.set_yticklabels(features, rotation=0)
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig(stem + '_corr_plot.png')

def plot_pca(X, save=False):
    cov = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    tot = sum(eigen_values)
    var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.bar(range(X.shape[1]), var_exp, alpha=0.5, align='center', label='Individual variance explained')
    ax.step(range(X.shape[1]), cum_var_exp, where='mid', label='Cumulative variance explained')
    ax.legend(loc='best')
    ax.axis('tight')
    ax.set_xlabel('n_components')
    ax.set_ylabel('explained variance ratio')
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig('pca_plot.png')

def plot_feature_importance(X, y, features, save=False):
    forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.set_title("Feature importance - Trees (Non-Parametric)")
    ax.bar(x=features[indices], height=importances[indices],
            color="r", yerr=std[indices], align="center")
    ax.set_xticklabels(features[indices], rotation=45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig('iportance-trees.png')

    f_test, _ = f_classif(X, y)
    f_test /= np.max(f_test)
    indices_ft = np.argsort(f_test)[::-1]

    mi = mutual_info_classif(X, y)
    mi /= np.max(mi)
    indices_mi = np.argsort(mi)[::-1]

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.set_title("Feature importance - Linear")
    ax.bar(x=features[indices_ft], height=f_test[indices_ft],
            color="r", align="center")
    ax.set_xticklabels(features[indices_ft], rotation=45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig('iportance-linear.png')

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.set_title("Feature importance - Mutual Information (Non-Linear)")
    ax.bar(x=features[indices_mi], height=mi[indices_mi],
            color="r", align="center")
    ax.set_xticklabels(features[indices_mi], rotation=45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig('iportance-mi.png')

    importances /= np.max(importances)
    comb_imp = importances + mi + f_test
    comb_imp /= np.max(comb_imp)
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    indices_ci = np.argsort(comb_imp)[::-1]

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.set_title("Feature importance - Combined")
    ax.bar(x=features[indices_ci], height=comb_imp[indices_ci],
            color="r", align="center")
    ax.set_xticklabels(features[indices_ci], rotation=45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    if save:
        fig.savefig('iportance-ci.png')

    return indices_ci



def get_class_weights(y):
    y = np.asarray(y)
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced'
                                                     , np.unique(y.reshape(y.shape[0], ))
                                                     , y.reshape(y.shape[0], ))
    return {cls: float(weight) for cls, weight in zip(np.unique(y.reshape(y.shape[0], )), class_weight)}


def roc_analysis_full(Y_test, Y_scores, classes=(1, 2, 3, 4)):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # Compute ROC curve and ROC area for each class
    Y_test = label_binarize(Y_test, classes=classes)  # make categorical
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(20, 15))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'black'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()