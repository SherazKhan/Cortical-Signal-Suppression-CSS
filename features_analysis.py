import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils import read_data, pca_diagnostics, plot_feature_corr, plot_pca, plot_feature_importance, autolabel
plt.ion()
plt.rcParams.update({'font.size': 22})

X_csv = 'data/DES009_Night4_SpindleFeatures_X.csv'
y_csv = 'data/DES009_Night4_Class_Y.csv'
X9_1, y9_1, features, scaler9_1 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X_csv = 'data/DES009_Night5_SpindleFeatures_X.csv'
y_csv = 'data/DES009_Night5_Class_Y.csv'
X9_2, y9_2, features, scaler9_2 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X_csv = 'data/DES004_Night8_SpindleFeatures_X.csv'
y_csv = 'data/DES004_Night8_Class_Y.csv'
X4_1, y4_1, features, scaler4_1 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X_csv = 'data/DES004_Night9_SpindleFeatures_X.csv'
y_csv = 'data/DES004_Night9_Class_Y.csv'
X4_2, y4_2, features, scaler4_1 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X_csv = 'data/DES012_Night1_SpindleFeatures_X.csv'
y_csv = 'data/DES012_Night1_Class_Y.csv'
X12_1, y12_1, features, scaler12_1 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X_csv = 'data/DES012_Night2_SpindleFeatures_X.csv'
y_csv = 'data/DES012_Night2_Class_Y.csv'
X12_2, y12_2, features, scaler12_1 = read_data(X_csv, y_csv, drop_columns=['startSample', 'endSample'], scale=True)

X = np.concatenate((X9_1, X9_2, X4_1, X4_2, X12_1, X12_2), axis=0)
y = np.concatenate((y9_1, y9_2, y4_1, y4_2, y12_1, y12_2))
X, y = shuffle(X, y)

#Class Distribution
classes= ['NOCS','SOCS','RICS','TRICS']
classes_prevalence = [(y==1).sum()*100/len(y), (y==2).sum()*100/len(y), (y==3).sum()*100/len(y), (y==4).sum()*100/len(y)]
fig, ax = plt.subplots(figsize=(14, 10))
rects = ax.bar(classes, classes_prevalence)
autolabel(rects, ax)
ax.set_xlabel('Class')
ax.set_ylabel('Prevalence')
fig.tight_layout()


classes= ['NOCS','SOCS','RICS','TRICS']
classes_prevalence = [(y==1).sum(), (y==2).sum(), (y==3).sum(), (y==4).sum()]
fig, ax = plt.subplots(figsize=(14, 10))
rects = ax.bar(classes, classes_prevalence)
autolabel(rects, ax, '{:.0f}')
ax.set_xlabel('Class')
ax.set_ylabel('Samples')
fig.tight_layout()

#Diagnostics
#pca_diagnostics(X, n_components=2, n_feat=2)
plot_feature_corr(X, features, stem='Feature_Correlation')
plot_pca(X)
plot_feature_importance(X, y, features)


