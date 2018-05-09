import os

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.metrics import pairwise, classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold, learning_curve

from visualize.plot import matrix as plotmatrix


# pipeline_BASE = Pipeline([
#     # Compute chi2 kernels for all modalities
#     ('chi2', histograms.Chi2Kernel(gamma=1.)),
#     # ('plot_kernels', histograms.PlotKernels("/home/maurice/Downloads/kernels", annot_font=6)),
#
#     ('combine_kernels', histograms.KernelAggregator(weights=[1, 1, 1, 1])),
#     # ('plot_agg_kernel', histograms.PlotKernels("/home/maurice/Downloads/kernels", annot_font=4)),
#
#     # Classify
#     ('svm', svm.NuSVC(nu=0.35, kernel='precomputed')),
#     # ('svm', svm.SVC(C=1.0, kernel='precomputed', probability=False, cache_size=1000))
# ])


def majority_vote(y_preds):
    """
    Return the 0/1 prediction that appears most in `y_preds`. Return -1 if the
    predictions appear the same amount of times.

    Parameters
    ----------
    y_preds : array_like, shape (n_preds,)
        0/1 predictions.

    Returns
    -------
    pred : int (0, 1 or -1)
        Prediction with highest frequency. Returns -1 if the predictions are exactly even.
    """
    y_preds = np.asarray(y_preds)
    n_0, n_1, n_other = 0, 0, 0
    for pred in y_preds:
        _pred = int(pred)
        if _pred == 0: n_0 += 1
        elif _pred == 1: n_1 += 1
        else: n_other += 1
    return 0 if n_0 > n_1 else 1 if n_1 > n_0 else -1


def accuracy(y_preds, y_true):
    """Return the percentage of predictions that are equal to y_true."""
    y_preds = np.array(y_preds)
    y_true_int = int(y_true)
    n_0, n_1= 0, 0
    for pred in y_preds:
        _pred = int(pred)
        if _pred == 0: n_0 += 1
        elif _pred == 1: n_1 += 1
        else: quit('Bad input')

    if y_true_int == 0: acc = n_0 / len(y_preds)
    elif y_true_int == 1: acc = n_1 / len(y_preds)
    else: quit('Bad input')

    return acc


def roipreds_2_patientpreds(roinames, roi_preds, roi_y):
    """
    Compute patient predictions from roi predictions.

    Parameters
    ----------
    roinames : array, shape (num_rois,)
        ROI names.
    roi_preds : array, shape (num_rois, num_preds)
        ROI predictions.
    roi_y : array, shape (num_rois,)
        The targets (0/1 ground truth) for the ROIs from which the patient targets will be extracted. It is assumed
        that the first four characters of each ROI name identifies the ROI's patient. Consequently, all ROIs belonging
        to a specific patient must have the same label.

    Returns
    -------
    patient_preds : pandas.DataFrame
        The columns (in order) are 'patient', 'num_rois', 'y_true', 'y_pred', 'accuracy'. 'accuracy' contains
        the percentage of correct predictions (from ROIs) for each patient.
    """
    assert len(roinames) == len(roi_preds) == len(roi_y)

    # Create the patient labels from the ROI labels. All ROI labels for a given patient should be the same. The
    # patient is the first four characters of an ROI name.
    patients_y_true = {}
    for roiname, truelabel in zip(roinames, roi_y):
        patient = roiname[:4]
        if patient in patients_y_true and patients_y_true[patient] != truelabel:
            quit('All ROIs for a patient should have same y_true value')
        else:
            patients_y_true[patient] = truelabel

    # Aggregate all predictions from ROIs for each patient
    patient_preds = defaultdict(list)
    for roiname, preds in zip(roinames, roi_preds):
        patient_preds[roiname[:4]].extend(list(preds))

    # Count number of ROIs for each patient
    patient_rois = defaultdict(int)
    for roiname in roinames:
        patient_rois[roiname[:4]] += 1

    # Create pandas DataFrame of patient predictions
    data = {'patient': [], 'num_rois': [], 'y_true': [], 'y_pred': [], 'accuracy': []}
    for patient, roi_predictions in patient_preds.items():
        data['patient'].append(patient)
        data['num_rois'].append(patient_rois[patient])
        data['y_true'].append(patients_y_true[patient])
        data['y_pred'].append(_majority_vote(patient_preds[patient]))
        data['accuracy'].append(_accuracy(patient_preds[patient], patients_y_true[patient]))
    patient_df = pd.DataFrame(data=data, columns=['patient', 'num_rois', 'y_true', 'y_pred', 'accuracy']).\
        sort_values(by='patient', ascending=True).reset_index(drop=True)

    return patient_df


def cross_validation(histograms, roinames, y, n_iters, n_folds, weights=None, gamma=0.6, class_weights=False,
                     retpreds='patient'):
    """
    Perform k-fold cross validation.

    Parameters
    ----------
    histograms : array_like of 2-d arrays
        The shape for each histogram should be (num_rois, num_bins), where the first dimension (num_rois,)
        should be the same among all histograms.
    roinames : array of str, shape (num_rois,)
        ROI names.
    y : array, shape (num_rois,)
        0/1 ground truth labels. The labels of ROIs that have the same four character prefix must
        be the same (because they are assumed to be from the same patient).
    n_iters : int
        Number of iterations.
    n_folds : int
        Number of folds.
    weights : list of float, optional
        The weights for the histograms. If not None, the weights should have shape (len(histograms),). If None,
        each histogram will be weighted equally.
    gamma : float, optional
        Used to create chi2 kernels.
    class_weights : bool, optional
        If True, weight each ROI by the ration of abnormal/normal ROIs.
    retpreds : str, optional
        Can be eiter 'patient' or 'roi'. If 'patient' the patient predictions will be returned. If 'roi' the
        roi predictions will be returned.

    Returns
    -------
    preds : pandas.DataFrame
        Returns either the patient or roi predictions depending on `retpreds`. If `retpreds` is 'patient', a
        DataFrame with columns 'Patient', 'num_rois', 'y_true', 'y_pred' and 'accuracy' will be returned. If
        `retpreds` is 'roi', a DataFrame with columns 'roiname', 'y_true', 'y_pred' and 'accuracy' will be
        returned.
    """
    if len(set([len(hist) for hist in histograms] + [len(roinames), len(y)])) != 1:
        raise ValueError('All input must have same number of ROIs')
    if retpreds not in ['patient', 'roi']:
        raise ValueError('retpreds must be "patient" or "roi"')

    histograms = [np.asarray(hist) for hist in histograms]
    num_rois = histograms[0].shape[0]

    if weights is None:
        weights = [1]*len(histograms)
    weights = np.array(weights, dtype=np.float)
    weights = weights / np.sum(weights)

    # Create pairwise chi2 kernels for each histogram and then aggregate them into a single kernel.
    K = np.zeros(shape=(num_rois, num_rois))
    for hist, weight in zip(histograms, weights):
        chi2 = pairwise.chi2_kernel(hist, gamma=gamma)
        mu = 1.0 / chi2.mean()
        K += (weight * np.exp(-mu * chi2))
        # K += chi2 * weight

    # clf = NuSVC(nu=0.35, kernel='precomputed')
    # plot_learning_curve(clf, "Learning Curves (gamma={})".format(gamma), K, y, train_sizes=np.linspace(0.2, 1.0, 30), cv=StratifiedKFold(n_splits=10))

    roi_preds = defaultdict(list)

    groups = np.array([roiname[:4] for roiname in roinames])

    # skf = StratifiedShuffleSplit(n_folds)
    # skf = GroupKFold(n_folds)
    # Cross validation
    for i in range(n_iters):
        skf = StratifiedKFold(n_folds, shuffle=True)
        for train, test in skf.split(K, y, groups):
            clf = NuSVC(nu=0.35, kernel='precomputed')
            K_train, K_test = K[train][:, train], K[test][:, train]
            y_train, y_test = y[train], y[test]

            plotmatrix(pd.DataFrame(data=K_train, index=roinames[train]),
                                  os.path.join('/home/maurice/Downloads/svmoutput', 'kernel_{}.pdf'.format(i)))

            sample_weights = None
            if class_weights:
                sample_weights = len(y_train) / (2 * np.bincount(y_train, minlength=2))
                if np.any(np.isinf(sample_weights)):
                    quit('Infinity sample weight. This means that all of the data has the same label')
                sample_weights = np.array([sample_weights[val] for val in y_train])

            clf.fit(K_train, y_train, sample_weights)

            y_pred = clf.predict(K_test)
            for roiname, pred in zip(roinames[test], y_pred):
                roi_preds[roiname].append(pred)

            print(classification_report(y_test, y_pred, [0, 1], ['Normal', 'Abnormal']), end='\n\n')

    if retpreds == 'roi':
        return pd.DataFrame({'roiname': roinames,
                             'y_true': y,
                             'y_pred': [_majority_vote(roi_preds[name]) for name in roinames],
                             'accuracy': [_accuracy(roi_preds[name], y_true) for name, y_true in zip(roinames, y)]},
                            columns=['roiname', 'y_true', 'y_pred', 'accuracy'])
    else:
        return roipreds_2_patientpreds(roinames, np.array([roi_preds[name] for name in roinames]), y)
