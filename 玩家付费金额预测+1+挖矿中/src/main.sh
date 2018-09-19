#!/bin/bash -x  

python _1_gbdt1_model.py
python _1_gbdt2_model.py
python _2_nn_model.py
python _2_gbdt_model.py
python _2_knn_lasso_model.py
python _2_lgb_model.py

python _3_merge.py

