#!/usr/bin/python
# -*- coding: latin-1 -*-
# ----------------------- IMPORTS ----------------------- #
import numpy as np
import os, sys, getopt, timeit, warnings

# ----------------------- IMPORTS ----------------------- #
def binary_performance_measure(y_true, y_pred):
    ''' y_true and y_pred must be NumPy arrays'''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # True Positives
    for i in range(y_true.shape[0]):
        if (y_true[i] == y_pred[i] == 1):
            TP += 1
    # False Positives
    for i in range(y_true.shape[0]):
        if (y_pred[i] == 1 and y_true[i] != y_pred[i]):
            FP += 1
    # True Negatives
    for i in range(y_true.shape[0]):
        if (y_true[i] == y_pred[i] == 0):
            TN += 1
    # False Begatives
    for i in range(y_true.shape[0]):
        if (y_pred[i] == 0 and y_true[i] != y_pred[i]):
            FN += 1
    return TP, FP, TN, FN

# Sensitivity, Recall or True Positive Rate
def Sensitivity(y_true, y_pred):
    TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
    return float(TP) / float(TP+FN)

# Specificity or True Negative Rate
def Specificity(y_true, y_pred):
    TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
    return float(TN) / float(TN+FP)

# Product of Sensitivity (Se) and Specificity (Sp)
def SexSp(y_true, y_pred):
    TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
    return (float(TP) / float(TP+FN)) * (float(TN) / float(TN+FP))

# Accuracy
def Accuracy(y_true, y_pred):
    TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
    return float(TP+TN) / float(TP+TN+FP+FN)






#
