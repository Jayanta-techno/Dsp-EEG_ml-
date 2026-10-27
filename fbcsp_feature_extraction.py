import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data


BANDS = [(8,12), (12,20), (20,30)]  #stable cross-subject bands


def fbcsp_features_cross_subject(X_train, y_train, X_test, fs=250, n_csp=6, bands=BANDS):

    feat_train = []
    feat_test = []
    csp_models = []

    for (l, h) in bands:
        #bandpass filter train/test to this band
        Xtr_f = filter_data(X_train, sfreq=fs, l_freq=l, h_freq=h, verbose=False)
        Xte_f = filter_data(X_test,  sfreq=fs, l_freq=l, h_freq=h, verbose=False)

        #CSP with regularization for stability across subjects
        csp = CSP(n_components=n_csp, reg="ledoit_wolf", log=True, norm_trace=False)

        #extract CSP log-variance features
        Ft = csp.fit_transform(Xtr_f, y_train)
        Fs = csp.transform(Xte_f)

        #store
        feat_train.append(Ft)
        feat_test.append(Fs)
        csp_models.append(csp)

    #concatenate features from all bands
    feat_train = np.concatenate(feat_train, axis=1)  # (Ntrain, bands*n_csp)
    feat_test  = np.concatenate(feat_test,  axis=1)  # (Ntest,  bands*n_csp)

    return feat_train, feat_test, csp_models
