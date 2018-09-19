import gc
import numpy as np
import pandas as pd

import os
import pickle
import lightgbm as lgb
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

def read_data():
    train_df = pd.read_csv('../data/tap_fun_train.csv')
    test_df = pd.read_csv('../data/tap_fun_test.csv')

    test_df["prediction_pay_price"] = -1
    test_usid = test_df.user_id
    data = pd.concat([train_df, test_df])
    del train_df, test_df
    gc.collect()

    return data

def add_SumZeros(data, features):
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    if 'SumZeros' in features:
        data.insert(1, 'SumZeros', (data[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    return data

def add_sumValues(data, features):
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    if 'SumValues' in features:
        data.insert(1, 'SumValues', (data[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    return data

def km(data):
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    scaler = StandardScaler()
    km_data = scaler.fit_transform(data[flist].values)

    flist_kmeans = []
    for ncl in range(5,15):
        clf = KMeans(n_clusters=ncl)
        clf.fit_predict(normalize(km_data, axis=0))
        data['kmeans_cluster_'+str(ncl)] = clf.predict(normalize(km_data, axis=0))
        flist_kmeans.append('kmeans_cluster_'+str(ncl))
    print(flist_kmeans)
    return data

def pca_pro(data):
    flist = [x for x in data.columns if not x in ['user_id','prediction_pay_price']]
    n_components = 40
    flist_pca = []
    pca = PCA(n_components=n_components)
    x_train_projected = pca.fit_transform(normalize(data[flist], axis=0))
    for npca in range(0, n_components):
        data.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
        flist_pca.append('PCA_'+str(npca+1))
    print(flist_pca)
    return data

def process(data):
    import time
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= 3600
    data['regedit_diff'] = (a - min(a))
    
    #
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    new['date'] = new['date'].apply(lambda x:x.split('-')[2])
    new['date_week'] = new.date.apply(lambda x:1 if x in ['27','28','03', '04','10','11','17','18','24','25'] else 0)
    data = pd.merge(data, new[['date_week', 'user_id']], on='user_id',how='left')
    
    #
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    new['date'] = new['date'].apply(lambda x:x.split('-')[2])
    new['date_holiday'] = new.date.apply(lambda x:1 if x in ['14','15','16'] else 0)
    data = pd.merge(data, new[['date_holiday', 'user_id']], on='user_id',how='left')
    
    #
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[1])
    new['date'] = new['date'].apply(lambda x:int(x.split(':')[0]))
    new['date_h_2'] = new.date.apply(lambda x:1 if ((x >= 4) & (x < 8) ) else 0)
    new['date_h_3'] = new.date.apply(lambda x:1 if ((x >= 8) & (x < 12) ) else 0)
    data = pd.merge(data, new[['date_h_2','date_h_3','user_id']], on='user_id',how='left')
    del data['register_time']
    
    #做一些比例的组合特征
    #每次充钱的比例
    data['pay_price_ave'] = data['pay_price'] / data['pay_count']
    data['pay_price_ave'] = data['pay_price_ave'].fillna(0)
    #副本赢得比例
    data['pve_win_ave'] = data['pve_win_count'] / data['pve_battle_count']
    data['pve_win_ave'] = data['pve_win_ave'].fillna(0)
    #主动发起副本的次数比例
    data['pve_lanch_ave'] = data['pve_lanch_count'] / data['pve_battle_count']
    data['pve_lanch_ave'] = data['pve_lanch_ave'].fillna(0)
    #人赢得比例
    data['pvp_win_ave'] = data['pvp_win_count'] / data['pvp_battle_count']
    data['pvp_win_ave'] = data['pvp_win_ave'].fillna(0)
    #主动发起与人战争的次数比例
    data['pvp_lanch_ave'] = data['pvp_lanch_count'] / data['pvp_battle_count']
    data['pvp_lanch_ave'] = data['pvp_lanch_ave'].fillna(0)
    
    data = add_SumZeros(data, ['SumZeros'])
    data = add_sumValues(data, ['SumValues'])
    
    data = km(data)
    data = pca_pro(data)

    return data


def split_with_payPrice(data):
    train = data.loc[data.prediction_pay_price != -1]
    test = data[data.prediction_pay_price == -1]

    test_with_pay_usid = test[test.pay_price != 0].user_id
    test_usid = data[data.prediction_pay_price == -1].user_id
    del test['user_id']
    del test['prediction_pay_price']
    test_X = test[test.pay_price != 0].values.astype(np.float32)

    del train['user_id']
    # y = np.log1p(train['prediction_pay_price'].values)
    y = train[train.pay_price != 0]['prediction_pay_price'].values.astype(np.float32)
    del train['prediction_pay_price']
    X = train[train.pay_price != 0].values.astype(np.float32)
    # col = train.columns

    del train, test
    gc.collect()
    return X, y, test_X, test_with_pay_usid, test_usid


def lgb_train(space, X_train, y_train, X_test):
    lgb_params ={'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 'metric': {'rmse'},
                 'num_leaves': space['num_leaves'], 'learning_rate': space['learning_rate'], 'max_bin': space['max_bin'], 
                 'max_depth': space['max_depth'], 'min_child_samples':space['min_child_samples'], 'subsample': space['subsample'],
                 'colsample_bytree': space['colsample_bytree'], 'nthread':4, 'verbose': 0}
    lgbtrain = lgb.Dataset(X_train, label=y_train)
    lgbtrain.construct()
    lgb_model = lgb.train(lgb_params, lgbtrain, num_boost_round=space['num_boost_round'])
    preds = lgb_model.predict(X_test, num_iteration=space['num_boost_round'])
    return lgb_model, preds


if __name__ == '__main__':
    train_data_path = '../data/lgb_train_data.pkl'
    if os.path.exists(train_data_path):
        X, y, test_X, test_with_pay_usid, test_usid = pickle.load(open(train_data_path,'rb'))
    else:
        data = read_data()
        data = process(data)
        X, y, test_X, test_with_pay_usid, test_usid = split_with_payPrice(data)

        pickle.dump((X, y, test_X, test_with_pay_usid, test_usid),open(train_data_path,'wb'))

    y = np.log1p(y)
    print("X, test_X:",X.shape, test_X.shape)

    lgb_best_params = {'colsample_bytree': 0.855, 
                        'learning_rate': 0.1, 
                        'max_bin': 428, 
                        'max_depth': 10,
                        'min_child_samples': 8, 
                        'num_boost_round': 538, 
                        'num_leaves': 53, 
                        'subsample': 0.91}

    lgb_model, preds = lgb_train(lgb_best_params, X, y, test_X)
    test_with_pay = pd.DataFrame()
    test_with_pay['user_id'] = test_with_pay_usid
    preds[preds < 0] = 0
    test_with_pay['prediction_pay_price'] = np.expm1(preds) * 1.49

    sub = pd.DataFrame()
    sub['user_id'] = test_usid
    sub['prediction_pay_price'] = 0
    sub.loc[sub.user_id.isin(test_with_pay.user_id), 'prediction_pay_price'] = test_with_pay['prediction_pay_price']

    print(sub.head(), '\n')
    print(sub.describe())
    sub.to_csv('./output/first_hyperopt_model_prediction.csv', index=False)



