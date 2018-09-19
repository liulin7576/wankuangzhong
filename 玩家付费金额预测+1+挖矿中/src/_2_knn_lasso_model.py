import gc
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold,cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Lasso,LassoCV,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
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

    #只取pay_price不为0的用户
    data = data[data.pay_price != 0]
    #添加注册用户的注册时间为周几的特征
    new = data[['user_id', 'register_time']]
    new['register_time'] = pd.to_datetime(new['register_time'])
    new['weekday'] = new['register_time'].apply(lambda x:x.weekday())
    data = pd.merge(data, new[['user_id', 'weekday']], how = 'left', on = 'user_id')

    return data ,test_usid


def get_poly_fea(data):
    fea_list = ['ivory_add_value', 'wood_add_value', 'stone_add_value', 'general_acceleration_add_value', 'ivory_reduce_value', 'meat_add_value', 'wood_reduce_value', 
            'training_acceleration_add_value']
    for i in fea_list:
        data[i + str(i) + 'sqrt'] = data[i] ** 0.5
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

def process(data):    
    data["wood_sub_value_abs"] = np.abs(data["wood_add_value"] - data["wood_reduce_value"])
    data["stone_sub_value_abs"] = np.abs(data["stone_add_value"] - data["stone_reduce_value"])
    data["ivory_sub_value_abs"] = np.abs(data["ivory_add_value"] - data["ivory_reduce_value"])
    data["meat_sub_value_abs"] = np.abs(data["meat_add_value"] - data["meat_reduce_value"])
    data["magic_sub_value_abs"] = np.abs(data["magic_add_value"] - data["magic_reduce_value"])
    data["infantry_sub_value_abs"] = np.abs(data["infantry_add_value"] - data["infantry_reduce_value"])
    data["cavalry_sub_value_abs"] = np.abs(data["cavalry_add_value"] - data["cavalry_reduce_value"])
    data["shaman_sub_value_abs"] = np.abs(data["shaman_add_value"] - data["shaman_reduce_value"])
    data["wound_infantry_sub_value_abs"] = np.abs(data["wound_infantry_add_value"] - data["wound_infantry_reduce_value"])
    data["wound_cavalry_sub_value_abs"] = np.abs(data["wound_cavalry_add_value"] - data["wound_cavalry_reduce_value"])
    data["wound_shaman_sub_value_abs"] = np.abs(data["wound_shaman_add_value"] - data["wound_shaman_reduce_value"])
    data["general_acceleration_sub_value_abs"] = np.abs(data["general_acceleration_add_value"] - data["general_acceleration_reduce_value"])
    data["building_acceleration_sub_value_abs"] = np.abs(data["building_acceleration_add_value"] - data["building_acceleration_reduce_value"])
    data["reaserch_acceleration_sub_value_abs"] = np.abs(data["reaserch_acceleration_add_value"] - data["reaserch_acceleration_reduce_value"])
    data["training_acceleration_sub_value_abs"] = np.abs(data["training_acceleration_add_value"] - data["training_acceleration_reduce_value"])
    data["treatment_acceleration_sub_value_abs"] = np.abs(data["treatment_acceleraion_add_value"] - data["treatment_acceleration_reduce_value"])
    data["bd_mean"] = data.iloc[:,34:50].mean(1)
    data["bd_std"] = data.iloc[:,34:50].std(1)
    data["bd_max"] = data.iloc[:,34:50].max(1)
    data["bd_min"] = data.iloc[:,34:50].min(1)
    data["sr_mean"] = data.iloc[:,50:99].mean(1)
    data["sr_std"] = data.iloc[:,50:99].std(1)
    data["sr_max"] = data.iloc[:,50:99].max(1)
    data["sr_min"] = data.iloc[:,50:99].min(1)
    data["source_add_value_mean"] = data.iloc[:,2:11:2].mean(1)
    data["source_add_value_std"] = data.iloc[:,2:11:2].std(1)
    data["source_add_value_max"] = data.iloc[:,2:11:2].max(1)
    data["source_add_value_min"] = data.iloc[:,2:11:2].min(1)
    data["source_reduce_value_mean"] = data.iloc[:,3:12:2].mean(1)
    data["source_reduce_value_std"] = data.iloc[:,3:12:2].std(1)
    data["source_reduce_value_max"] = data.iloc[:,3:12:2].max(1)
    data["source_reduce_value_min"] = data.iloc[:,3:12:2].min(1)
    data["military_add_value_mean"] = data.iloc[:,12:17:2].mean(1)
    data["military_add_value_std"] = data.iloc[:,12:17:2].std(1)
    data["military_add_value_max"] = data.iloc[:,12:17:2].max(1)
    data["military_add_value_min"] = data.iloc[:,12:17:2].min(1)
    data["military_reduce_value_mean"] = data.iloc[:,13:18:2].mean(1)
    data["military_reduce_value_std"] = data.iloc[:,13:18:2].std(1)
    data["military_reduce_value_max"] = data.iloc[:,13:18:2].max(1)
    data["military_reduce_value_min"] = data.iloc[:,13:18:2].min(1)
    data["wound_add_value_mean"] = data.iloc[:,18:23:2].mean(1)
    data["wound_add_value_std"] = data.iloc[:,18:23:2].std(1)
    data["wound_add_value_max"] = data.iloc[:,18:23:2].max(1)
    data["wound_add_value_min"] = data.iloc[:,18:23:2].min(1)
    data["wound_reduce_value_mean"] = data.iloc[:,19:24:2].mean(1)
    data["wound_reduce_value_std"] = data.iloc[:,19:24:2].std(1)
    data["wound_reduce_value_max"] = data.iloc[:,19:24:2].max(1)
    data["wound_reduce_value_min"] = data.iloc[:,19:24:2].min(1)
    data["acceleration_add_value_mean"] = data.iloc[:,24:33:2].mean(1)
    data["acceleration_add_value_std"] = data.iloc[:,24:33:2].std(1)
    data["acceleration_add_value_max"] = data.iloc[:,24:33:2].max(1)
    data["acceleration_add_value_min"] = data.iloc[:,24:33:2].min(1)
    data["acceleration_reduce_value_mean"] = data.iloc[:,25:34:2].mean(1)
    data["acceleration_reduce_value_std"] = data.iloc[:,25:34:2].std(1)
    data["acceleration_reduce_value_max"] = data.iloc[:,25:34:2].max(1)
    data["acceleration_reduce_value_min"] = data.iloc[:,25:34:2].min(1)
    data["atk_level_mean"] = data[["sr_infantry_atk_level","sr_cavalry_atk_level","sr_shaman_atk_level","sr_troop_attack_level"]].mean(1)
    data["atk_level_std"] = data[["sr_infantry_atk_level","sr_cavalry_atk_level","sr_shaman_atk_level","sr_troop_attack_level"]].std(1)
    data["atk_level_max"] = data[["sr_infantry_atk_level","sr_cavalry_atk_level","sr_shaman_atk_level","sr_troop_attack_level"]].max(1)
    data["atk_level_min"] = data[["sr_infantry_atk_level","sr_cavalry_atk_level","sr_shaman_atk_level","sr_troop_attack_level"]].min(1)
    data["def_level_mean"] = data[["sr_infantry_def_level","sr_cavalry_def_level","sr_shaman_def_level","sr_troop_defense_level"]].mean(1)
    data["def_level_std"] = data[["sr_infantry_def_level","sr_cavalry_def_level","sr_shaman_def_level","sr_troop_defense_level"]].std(1)
    data["def_level_max"] = data[["sr_infantry_def_level","sr_cavalry_def_level","sr_shaman_def_level","sr_troop_defense_level"]].max(1)
    data["def_level_min"] = data[["sr_infantry_def_level","sr_cavalry_def_level","sr_shaman_def_level","sr_troop_defense_level"]].min(1)
    data["hp_level_mean"] = data[["sr_infantry_hp_level","sr_cavalry_hp_level","sr_shaman_hp_level"]].mean(1)
    data["hp_level_std"] = data[["sr_infantry_hp_level","sr_cavalry_hp_level","sr_shaman_hp_level"]].std(1)
    data["hp_level_max"] = data[["sr_infantry_hp_level","sr_cavalry_hp_level","sr_shaman_hp_level"]].max(1)
    data["hp_level_min"] = data[["sr_infantry_hp_level","sr_cavalry_hp_level","sr_shaman_hp_level"]].min(1)
    data["sr_prod_level_mean"] = data[["sr_rss_a_prod_levell","sr_rss_b_prod_level","sr_rss_c_prod_level","sr_rss_d_prod_level"]].mean(1)
    data["sr_prod_level_std"] = data[["sr_rss_a_prod_levell","sr_rss_b_prod_level","sr_rss_c_prod_level","sr_rss_d_prod_level"]].std(1)
    data["sr_prod_level_max"] = data[["sr_rss_a_prod_levell","sr_rss_b_prod_level","sr_rss_c_prod_level","sr_rss_d_prod_level"]].max(1)
    data["sr_prod_level_min"] = data[["sr_rss_a_prod_levell","sr_rss_b_prod_level","sr_rss_c_prod_level","sr_rss_d_prod_level"]].min(1)
    data["sr_gather_level_mean"] = data[["sr_rss_a_gather_level","sr_rss_b_gather_level","sr_rss_c_gather_level","sr_rss_d_gather_level"]].mean(1)
    data["sr_gather_level_std"] = data[["sr_rss_a_gather_level","sr_rss_b_gather_level","sr_rss_c_gather_level","sr_rss_d_gather_level"]].std(1)
    data["sr_gather_level_max"] = data[["sr_rss_a_gather_level","sr_rss_b_gather_level","sr_rss_c_gather_level","sr_rss_d_gather_level"]].max(1)
    data["sr_gather_level_min"] = data[["sr_rss_a_gather_level","sr_rss_b_gather_level","sr_rss_c_gather_level","sr_rss_d_gather_level"]].min(1)

    import time
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= (3600 * 24)  #这里试试天数会不会更有效果
    data['regedit_diff_day'] = (a - min(a))
    
    data['regedit_diff_week'] = a.apply(lambda x:int(x)) % 7
    
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= 3600
    data['regedit_diff_hour'] = (a - min(a))
    
    #
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-02-02', '2018-02-09', '2018-02-16', '2018-02-23', '2018-03-02','2018-03-09','2018-03-16']
    new['date_week'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_week', 'user_id']], on='user_id',how='left')
    
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-02-13', '2018-03-16']
    new['date_week_two'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_week_two', 'user_id']], on='user_id',how='left')

    #
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-03-10', '2018-02-19']
    new['date_holiday'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_holiday', 'user_id']], on='user_id',how='left')
    
    ###
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[1])
    new['date'] = new['date'].apply(lambda x:int(x.split(':')[0]))
    new['date_h_1'] = new.date.apply(lambda x:1 if ((x >= 0) & (x < 4) )else 0)
    new['date_h_2'] = new.date.apply(lambda x:1 if ((x >= 4) & (x < 8) )else 0)
    new['date_h_3'] = new.date.apply(lambda x:1 if ((x >= 8) & (x < 12) )else 0)
    new['date_h_4'] = new.date.apply(lambda x:1 if ((x >= 12) & (x < 16) )else 0)
    new['date_h_5'] = new.date.apply(lambda x:1 if ((x >= 16) & (x < 20) )else 0)
    new['date_h_6'] = new.date.apply(lambda x:1 if ((x >= 20) & (x < 24) )else 0)
    data = pd.merge(data, new[['date_h_2','date_h_3','user_id']], on='user_id',how='left')
    
    data['register_time'] = pd.to_datetime(data['register_time'])
    data['dow'] = data['register_time'].apply(lambda x:x.dayofweek)
    data['doy'] = data['register_time'].apply(lambda x:x.dayofyear)
    data['day'] = data['register_time'].apply(lambda x:x.day)
    data['month'] = data['register_time'].apply(lambda x:x.month)
    data['hour'] = data['register_time'].apply(lambda x:x.hour)
    data['minute'] = data['register_time'].apply(lambda x:x.hour*60 + x.minute)
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
    
    #在线时长和PVE
    data['pve_battle_count_min'] = data['pve_battle_count'] / data['avg_online_minutes']
    data.loc[data.avg_online_minutes == 0, 'pve_battle_count_min'] = 0
    data['pve_battle_count_min'] = data['pve_battle_count_min'].fillna(0)
    #在线时长和主动PVE
    data['pve_lanch_count_min'] = data['pve_lanch_count'] / data['avg_online_minutes']
    data.loc[data.avg_online_minutes == 0, 'pve_lanch_count_min'] = 0
    data['pve_lanch_count_min'] = data['pve_lanch_count_min'].fillna(0)

    #在线时长和PVp
    data['pvp_battle_count_min'] = data['pvp_battle_count'] / data['avg_online_minutes']
    data.loc[data.avg_online_minutes == 0, 'pvp_battle_count_min'] = 0
    data['pvp_battle_count_min'] = data['pvp_battle_count_min'].fillna(0)
    #在线时长和主动PVp
    data['pvp_lanch_count_min'] = data['pvp_lanch_count'] / data['avg_online_minutes']
    data.loc[data.avg_online_minutes == 0, 'pvp_lanch_count_min'] = 0
    data['pvp_lanch_count_min'] = data['pvp_lanch_count_min'].fillna(0)
    
    data = get_poly_fea(data)
    data = add_SumZeros(data, ['SumZeros'])
    data = add_sumValues(data, ['SumValues'])
    data = km(data)

    return data


def split(data):
    train = data[data.prediction_pay_price != -1]
    test = data[data.prediction_pay_price == -1]
    y = train['prediction_pay_price'].values.astype(np.float32)    
    numeric_feats = data.drop(['prediction_pay_price', 'user_id'],axis=1).dtypes[\
                    data.drop(['prediction_pay_price', 'user_id'],axis=1).dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    print(skewed_feats.shape)

    skewed_feats = skewed_feats.index
    train[skewed_feats] = np.log1p(train[skewed_feats])
    test[skewed_feats] = np.log1p(test[skewed_feats])
    test_usid = test.user_id

    del test['user_id']
    del test['prediction_pay_price']

    test_X = test.values.astype(np.float32)
    del train['user_id']
    del train['prediction_pay_price']

    X = train.values.astype(np.float32)

    return X, y, test_X, test_usid
    
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


if __name__ == '__main__':

    train_data_path = '../data/knn_lasso_data.pkl'
    if os.path.exists(train_data_path):
        data ,test_usid = pickle.load(open(train_data_path,'rb'))

    else:
        data ,test_usid = read_data()
        data = process(data)
        pickle.dump((data,test_usid), open(train_data_path,'wb'))

    test_with_pay = pd.DataFrame()
    test_with_pay['user_id'] = data[data.prediction_pay_price == -1].user_id
    test_with_pay['prediction_pay_price'] = 0

    flist = [5.98, 6.97, 10.98, 16.96, 11.97, 9.99, 26.95, 21.95, 3.96, 11.96]
    for i in list(data.pay_price.value_counts().index):
        if (len(data[data.pay_price == i]) > 300) and (i not in flist):
            X, y, test_X, test_with_pay_usid  = split(data[data.pay_price == i])
            y = np.log1p(y)

            #gbdt加入新特征进去
            clf = xgb.XGBRegressor(
                n_estimators=30,#三十棵树
                learning_rate =0.1,
                max_depth=3,
                min_child_weight=1,
                gamma=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                reg_lambda=1,
                seed=27)

            model_sklearn=clf.fit(X, y)
            y_sklearn= clf.predict(test_X)
            print('max:',max(y_sklearn))

            train_new_feature= clf.apply(X)#每个样本在每颗树叶子节点的索引值
            test_new_feature= clf.apply(test_X)
            X = train_new_feature.copy()
            test_X = test_new_feature.copy()

            knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
            knn.fit(X, np.expm1(y))
            print('pay_price of i',i, 'rmse:', rmse_cv(knn).mean())

            knn_preds = knn.predict(test_X)
            knn_preds = np.where(knn_preds < 0 ,0.99, knn_preds)
            print("knn_preds max:",max(knn_preds))
            test_with_pay.loc[test_with_pay.user_id.isin(test_with_pay_usid), 'prediction_pay_price'] = knn_preds

    save_usid = []
    for i in list(data.pay_price.value_counts().index):
        if (len(data[data.pay_price == i]) <= 300) or (i in flist):
            save_usid += (data[data.pay_price == i].user_id.tolist())
            
    X, y, test_X, test_with_pay_usid  = split(data[data.user_id.isin(save_usid)])
    y = np.log1p(y)

    rsc = RobustScaler()
    X = rsc.fit_transform(X)
    test_X = rsc.transform(test_X)

    # xgb_regressor = xgb.XGBRegressor()
    model_lasso = LassoCV(alphas = [1, 0.1, 0.005,0.003,  0.001, 0.0005, 0.0001])
    sfm = SelectFromModel(model_lasso)
    sfm.fit(X, y)
    X = sfm.transform(X)
    test_X = sfm.transform(test_X)

    print("max y:",max(y))

    # model_lasso = LassoCV(alphas = [1, 0.1, 0.005,0.003,  0.001, 0.0005, 0.0001])
    # model_lasso = make_pipeline(RobustScaler(), model_lasso).fit(X, y)

    # maxOS 系统跑 lasso算法预测值异常，所以用ridge 代替了，如果复现结果差异较大的话，请尝试 model_lasso = Lasso()代替下面一行
    model_lasso = Ridge(alpha=1.0, max_iter=100, tol=0.001, random_state=24)
    model_lasso.fit(X, y)
    lasso_preds = model_lasso.predict(test_X)
    lasso_preds = np.where(lasso_preds < 0 ,0.99, lasso_preds)
    print("lasso_preds max:", max(lasso_preds))

    test_with_pay.loc[test_with_pay.user_id.isin(test_with_pay_usid), 'prediction_pay_price'] = np.expm1(lasso_preds)
    print(test_with_pay.describe())

    sub = pd.DataFrame()
    sub['user_id'] = test_usid.values
    sub['prediction_pay_price'] = 0
	#乘以1.432是为了保证训练集和测试集的prediction_pay_price均值大致相同
    sub.loc[sub.user_id.isin(test_with_pay.user_id), 'prediction_pay_price'] = test_with_pay['prediction_pay_price'].values * 1.432
    print(sub.head(), '\n')
    print(sub.describe())

    sub.to_csv('./output/knn_lasso.csv', index=False)


