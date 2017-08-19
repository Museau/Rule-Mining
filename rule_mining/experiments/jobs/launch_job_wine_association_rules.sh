# !/bin/bash

# Wine dataset
# Creat the directories if not exist
mkdir -p log_association_rules/wine

# Association rules + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_association_rules_L2LR' --file_name 'wine' --orm_type 'association_rules' > log_association_rules/wine/wine_test_association_rules_L2LR.txt 2> log_association_rules/wine/wine_erreur_test_association_rules_L2LR.txt

# Association rules + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_association_rules_L1LR' --file_name 'wine' --orm_type 'association_rules' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_association_rules/wine/wine_test_association_rules_L1LR.txt 2> log_association_rules/wine/wine_erreur_test_association_rules_L1LR.txt

# Association rules + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_association_rules_SVMlinear' --file_name 'wine' --orm_type 'association_rules' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_association_rules/wine/wine_test_association_rules_SVMlinear.txt 2> log_association_rules/wine/wine_erreur_test_association_rules_SVMlinear.txt

# Association rules + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_association_rules_SVMrbf' --file_name 'wine' --orm_type 'association_rules' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_association_rules/wine/wine_test_association_rules_SVMrbf.txt 2> log_association_rules/wine/wine_erreur_test_association_rules_SVMrbf.txt
