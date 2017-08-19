# !/bin/bash

# Balance scale dataset
# Creat the directories if not exist
mkdir -p log_decision_tree/balance_scale

# Decision tree + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_L2LR_du' --file_name 'balance_scale' --prepro_type 'dummy' --orm_type 'decision_tree' > log_decision_tree/balance_scale/balance_scale_test_decision_tree_L2LR_du.txt 2> log_decision_tree/balance_scale/balance_scale_erreur_test_decision_tree_L2LR_du.txt
# Decision tree + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_L1LR_du' --file_name 'balance_scale' --prepro_type 'dummy' --orm_type 'decision_tree' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_decision_tree/balance_scale/balance_scale_test_decision_tree_L1LR_du.txt 2> log_decision_tree/balance_scale/balance_scale_erreur_test_decision_tree_L1LR_du.txt
# Decision tree + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_SVMlinear_du' --file_name 'balance_scale' --prepro_type 'dummy' --orm_type 'decision_tree' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_decision_tree/balance_scale/balance_scale_test_decision_tree_SVMlinear_du.txt 2> log_decision_tree/balance_scale/balance_scale_erreur_test_decision_tree_SVMlinear_du.txt
# Decision tree + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_SVMrbf_du' --file_name 'balance_scale' --prepro_type 'dummy' --orm_type 'decision_tree' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_decision_tree/balance_scale/balance_scale_test_decision_tree_SVMrbf_du.txt 2> log_decision_tree/balance_scale/balance_scale_erreur_test_decision_tree_SVMrbf_du.txt
