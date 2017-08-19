# !/bin/bash

# Balance scale dataset
# Creat the directories if not exist
mkdir -p log_orm1d/balance_scale

# RM1D + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L2LR' --file_name 'balance_scale' --prepro_type 'None' > log_orm1d/balance_scale/balance_scale_test_ORM1D_L2LR.txt 2> log_orm1d/balance_scale/balance_scale_erreur_test_ORM1D_L2LR.txt

# RM1D + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L1LR' --file_name 'balance_scale' --prepro_type 'None' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_orm1d/balance_scale/balance_scale_test_ORM1D_L1LR.txt 2> log_orm1d/balance_scale/balance_scale_erreur_test_ORM1D_L1LR.txt

# RM1D + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMlinear' --file_name 'balance_scale' --prepro_type 'None' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/balance_scale/balance_scale_test_ORM1D_SVMlinear.txt 2> log_orm1d/balance_scale/balance_scale_erreur_test_ORM1D_SVMlinear.txt

# RM1D + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMrbf' --file_name 'balance_scale' --prepro_type 'None' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/balance_scale/balance_scale_test_ORM1D_SVMrbf.txt 2> log_orm1d/balance_scale/balance_scale_erreur_test_ORM1D_SVMrbf.txt
