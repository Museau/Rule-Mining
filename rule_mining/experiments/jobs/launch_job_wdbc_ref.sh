# !/bin/bash

# WDBC dataset
# Creat the directories if not exist
mkdir -p log_ref/wdbc

# Experience on discretized data
# Ref scores L2LR, L1LR, SVMlinear, SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_L2LR' --file_name 'wdbc' --orm_type 'None' --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_L2LR.txt 2> log_ref/wdbc/wdbc_erreur_test_L2LR.txt
python2 rule_mining/experiments/main.py --path_test 'test_L1LR' --file_name 'wdbc' --orm_type 'None' --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_ref/wdbc/wdbc_test_L1LR.txt 2> log_ref/wdbc/wdbc_erreur_test_L1LR.txt
python2 rule_mining/experiments/main.py --path_test 'test_SVMlinear' --file_name 'wdbc' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_SVMlinear.txt 2> log_ref/wdbc/wdbc_erreur_test_SVMlinear.txt
python2 rule_mining/experiments/main.py --path_test 'test_SVMrbf' --file_name 'wdbc' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_SVMrbf.txt 2> log_ref/wdbc/wdbc_erreur_test_SVMrbf.txt

# Ref scores RF, GBT
python2 rule_mining/experiments/random_forest_main.py --file_name 'wdbc' > log_ref/wdbc/wdbc_test_RF.txt 2> log_ref/wdbc/wdbc_erreur_test_RF.txt
python2 rule_mining/experiments/gradient_boosted_tree_main.py --file_name 'wdbc' > log_ref/wdbc/wdbc_test_GBT.txt 2> log_ref/wdbc/wdbc_erreur_test_GBT.txt


# Experience on non-discretized data
# Ref scores L2LR, L1LR, SVMlinear, SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_L2LR_nd' --file_name 'wdbc' --prepro_type 'None' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_L2LR_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_L2LR_nd.txt
python2 rule_mining/experiments/main.py --path_test 'test_L1LR_nd' --file_name 'wdbc' --prepro_type 'None' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_ref/wdbc/wdbc_test_L1LR_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_L1LR_nd.txt
python2 rule_mining/experiments/main.py --path_test 'test_SVMlinear_nd' --file_name 'wdbc' --prepro_type 'None' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_SVMlinear_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_SVMlinear_nd.txt
python2 rule_mining/experiments/main.py --path_test 'test_SVMrbf_nd' --file_name 'wdbc' --prepro_type 'None' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' --orm_type 'None'  --mod_size_threshold_init {} --size_threshold_init {} --purity_threshold_init {} --z_score_threshold_init {} --mod_size_threshold {} --size_threshold {} --purity_threshold {} --z_score_threshold {} > log_ref/wdbc/wdbc_test_SVMrbf_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_SVMrbf_nd.txt

# Ref scores RF, GBT
python2 rule_mining/experiments/random_forest_main.py --file_name 'wdbc' --prepro_type 'None' > log_ref/wdbc/wdbc_test_RF_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_RF_nd.txt
python2 rule_mining/experiments/gradient_boosted_tree_main.py --file_name 'wdbc' --prepro_type 'None' > log_ref/wdbc/wdbc_test_GBT_nd.txt 2> log_ref/wdbc/wdbc_erreur_test_GBT_nd.txt
