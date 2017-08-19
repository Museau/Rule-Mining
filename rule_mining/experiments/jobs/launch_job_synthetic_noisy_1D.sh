# !/bin/bash

# Synthetic noisy dataset
# Creat the directories if not exist
mkdir -p log_orm1d/synthetic_noisy

# RM1D + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L2LR_deltaTTT_d' --file_name 'synthetic_noisy' --prepro_type 'discretized' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' > log_orm1d/synthetic_noisy/synthetic_noisy_test_ORM1D_L2LR_deltaTTT_d.txt 2> log_orm1d/synthetic_noisy/synthetic_noisy_erreur_test_ORM1D_L2LR_deltaTTT_d.txt

# RM1D + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_L1LR_deltaTTT_d' --file_name 'synthetic_noisy' --prepro_type 'discretized' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_orm1d/synthetic_noisy/synthetic_noisy_test_ORM1D_L1LR_deltaTTT_d.txt 2> log_orm1d/synthetic_noisy/synthetic_noisy_erreur_test_ORM1D_L1LR_deltaTTT_d.txt

# RM1D + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMlinear_deltaTTT_d' --file_name 'synthetic_noisy' --prepro_type 'discretized' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/synthetic_noisy/synthetic_noisy_test_ORM1D_SVMlinear_deltaTTT_d.txt 2> log_orm1d/synthetic_noisy/synthetic_noisy_erreur_test_ORM1D_SVMlinear_deltaTTT_d.txt

# RM1D + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_ORM1D_SVMrbf_deltaTTT_d' --file_name 'synthetic_noisy' --prepro_type 'discretized' --local_feature_type 'distance_rule' --local_feature_params '{"wi": true, "wr": true, "centered": true}' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_orm1d/synthetic_noisy/synthetic_noisy_test_ORM1D_SVMrbf_deltaTTT_d.txt 2> log_orm1d/synthetic_noisy/synthetic_noisy_erreur_test_ORM1D_SVMrbf_deltaTTT_d.txt
