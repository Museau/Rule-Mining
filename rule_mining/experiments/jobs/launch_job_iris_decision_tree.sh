# !/bin/bash

# Iris dataset
# Creat the directories if not exist
mkdir -p log_decision_tree/iris

# Decision tree + L2LR
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_L2LR' --orm_type 'decision_tree' > log_decision_tree/iris/iris_test_decision_tree_L2LR.txt 2> log_decision_tree/iris/iris_erreur_test_decision_tree_L2LR.txt

# Decision tree + L1LR
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_L1LR' --orm_type 'decision_tree' --classifier_type 'LR' --classifier_params '{"penalty": ["l1"], "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "class_weight": ["balanced"], "random_state": [0], "multi_class": ["ovr"]}' > log_decision_tree/iris/iris_test_decision_tree_L1LR.txt 2> log_decision_tree/iris/iris_erreur_test_decision_tree_L1LR.txt

# Decision tree + SVMlinear
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_SVMlinear' --orm_type 'decision_tree' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["linear"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_decision_tree/iris/iris_test_decision_tree_SVMlinear.txt 2> log_decision_tree/iris/iris_erreur_test_decision_tree_SVMlinear.txt

# Decision tree + SVMrbf
python2 rule_mining/experiments/main.py --path_test 'test_decision_tree_SVMrbf' --orm_type 'decision_tree' --classifier_type 'SVM' --classifier_params '{"C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "kernel": ["rbf"], "probability": [true], "class_weight": ["balanced"], "decision_function_shape": ["ovr"], "random_state": [0]}' > log_decision_tree/iris/iris_test_decision_tree_SVMrbf.txt 2> log_decision_tree/iris/iris_erreur_test_decision_tree_SVMrbf.txt
