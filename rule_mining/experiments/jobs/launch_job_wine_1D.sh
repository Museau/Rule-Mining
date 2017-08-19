# !/bin/bash

# Wine dataset
# Creat the directories if not exist
mkdir -p log_orm1d/wine

# RM1D + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L2LR_deltaTTT' --file_name 'wine' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' > log_orm1d/wine/wine_test_ORM1D_L2LR_deltaTTT.txt 2> log_orm1d/wine/wine_erreur_test_ORM1D_L2LR_deltaTTT.txt

# RM1D + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L1LR_deltaTTT' --file_name 'wine' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_orm1d/wine/wine_test_ORM1D_L1LR_deltaTTT.txt 2> log_orm1d/wine/wine_erreur_test_ORM1D_L1LR_deltaTTT.txt

# RM1D + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMlinear_deltaTTT' --file_name 'wine' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/wine/wine_test_ORM1D_SVMlinear_deltaTTT.txt 2> log_orm1d/wine/wine_erreur_test_ORM1D_SVMlinear_deltaTTT.txt

# RM1D + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMrbf_deltaTTT' --file_name 'wine' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/wine/wine_test_ORM1D_SVMrbf_deltaTTT.txt 2> log_orm1d/wine/wine_erreur_test_ORM1D_SVMrbf_deltaTTT.txt
