import csv
import numpy as np
import helpers
import implementations_s as imp
import cross_validation as demo

""" 
We use Ridge Regression with parameters:
degree = 4
lambda = 0.013

We clean the data according to the subgroups defined by the value of the PRI_jet_num feature. Every non-defined feature is replaced by the subgroup mean of this feature.

The predictions are stored in the /submisssions file.
"""

print("Reading and cleaning the data...")

# reading the train data
DATA_TRAIN_PATH = '../../data/train.csv'
y, x, ids = helpers.load_csv_data(DATA_TRAIN_PATH)

# feature names and their respective indices
string_features = 'DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt'

features = string_features.split(",")
dict_ = {}
for ind, feat in enumerate(features):
    dict_[feat] = ind

# replacing the non-defined features by the mean of the features, according by the reactor
x_subgroups_list, ids_list = helpers.subgrouping(x, ids, dict_)
x_subgroups = helpers.group(x_subgroups_list, ids_list, dict_)

# standardize the data
y_clean = y
x_clean, _, _ = helpers.standardize(x_subgroups)

# reading the test data
DATA_TEST_PATH = '../../data/test.csv' 
OUTPUT_PATH = './submissions'
_, x_test, ids_test = helpers.load_csv_data(DATA_TEST_PATH)

# cleaning and standardize the test input
x_test_subgroups_list, ids_list = helpers.subgrouping(x_test, ids_test, dict_)
x_test_subgroups = helpers.group(x_test_subgroups_list, ids_list, dict_)
x_test_clean, _, _ = helpers.standardize(x_test_subgroups)

# augmenting the test input
tx_test = helpers.build_poly(x_test_clean, 4)
print("data ok")

# compute the model parameters
w, loss = imp.ridge_regression(y_clean, helpers.build_poly(x_clean, 4), lambda_=0.013)
print("Model parameters ok")

# predict and write predictions
y_pred = helpers.predict_labels(w, tx_test)
helpers.create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print("Predictions ok")