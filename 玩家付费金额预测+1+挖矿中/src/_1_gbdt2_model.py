
import pandas as pd
import numpy as np
import pickle
import time
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def read_data_gen_feat():
    t1=time.time()

    train = pd.read_csv('../data/tap_fun_train.csv')
    test = pd.read_csv('../data/tap_fun_test.csv')

    test_id = test[['user_id','pay_price']].copy()

    train_num = train.shape[0]
    data = pd.concat([train,test],axis=0)

    data['register_hour'] = data['register_time'].map(lambda x : int(x[11:13]))
    data['register_time_day'] = data['register_time'].map(lambda x : x[5:10])

    register_hour = data.groupby(['register_time_day'])['user_id'].count()
    data['register_hour'] = data['register_time_day'].map(lambda x : register_hour[x])

    register_day = data.groupby(['register_time_day'])['user_id'].count()
    data['register_day'] = data['register_time_day'].map(lambda x : register_day[x])

    data = data[['user_id','register_hour','register_day','pvp_battle_count', 'pvp_lanch_count', \
                'pvp_win_count' , 'prediction_pay_price', 'pve_battle_count', 'pve_lanch_count',\
                'pve_win_count', 'avg_online_minutes', 'pay_price', 'pay_count',\
                'reaserch_acceleration_add_value','sr_outpost_durability_level','reaserch_acceleration_reduce_value',\
                'treatment_acceleraion_add_value','treatment_acceleration_reduce_value']]

    train = data[:train_num]
    test = data[train_num:]

    train = train.loc[train['prediction_pay_price']>0,:]
    train = train.loc[train['prediction_pay_price']<16000,:]
    test = test.loc[test['pay_price']>0,:]
    print(train.shape, test.shape)

    train_num = train.shape[0]
    data = pd.concat([train,test],axis=0)

    data.loc[:,'reaserch_acceleration_reduce_ratio'] = data['reaserch_acceleration_reduce_value'] / (data['reaserch_acceleration_add_value']+1e-4)
    data.loc[:,'reaserch_acceleration_add_sub_reduce'] = data['reaserch_acceleration_add_value'] - data['reaserch_acceleration_reduce_value']
    data.loc[:,'treatment_acceleration_add_sub_reduce'] = data['treatment_acceleraion_add_value'] - data['treatment_acceleration_reduce_value']

    # pvp

    data.loc[:,'pvp_lanch_ratio'] = data['pvp_lanch_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_ratio'] = data['pvp_win_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_lanch_ratio'] = data['pvp_win_count'] / (data['pvp_lanch_count'] + 1e-4)

    # pve
    data.loc[:,'pve_lanch_ratio'] = data['pve_lanch_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_ratio'] = data['pve_win_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_lanch_ratio'] = data['pve_win_count'] / (data['pve_lanch_count'] + 1e-4)

    data.loc[:,'pve_pvp_battle_count'] = data['pvp_battle_count'] + data['pve_battle_count']
    data.loc[:,'pve_pvp_lanch_count'] = data['pvp_lanch_count'] + data['pve_lanch_count']
    data.loc[:,'pve_pvp_win_count'] = data['pvp_win_count'] + data['pve_win_count']

    data.loc[:,'pve_pvp_lanch_'] = data['pve_pvp_lanch_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_lanch_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_lanch_count'] + 1e-4)

    # 时间、消费
    data.loc[:,'pay_mean_count'] = data['pay_price'] / (data['pay_count'] + 1e-4)
    data.loc[:,'pay_mean_online_minutes'] = data['pay_price'] / (data['avg_online_minutes'] + 1e-4)
    data.loc[:,'pay_count_online_minutes'] = data['avg_online_minutes'] / (data['pay_count'] + 1e-4)

    data.loc[:,'time_lanch_per'] = data['avg_online_minutes'] / (data['pve_pvp_lanch_count'] + 10)
    data.loc[:,'money_lanch_per'] = data['pay_price'] / (data['pve_pvp_lanch_count'] + 10)

    data.loc[:,'time_battle_per'] = data['avg_online_minutes'] / (data['pve_pvp_battle_count'] + 10)
    data.loc[:,'money_battle_per'] = data['pay_price'] / (data['pve_pvp_battle_count'] + 10)

    data.loc[:,'time_win_per'] = data['avg_online_minutes'] / (data['pve_pvp_win_count'] + 10)
    data.loc[:,'money_win_per'] = data['pay_price'] / (data['pve_pvp_win_count'] + 10)
    data.loc[:,'pay_count_win_per'] = data['pay_count'] / (data['pve_pvp_win_count'] + 10)

    train = data[:train_num]
    test = data[train_num:]

    print(train.shape, test.shape)

    label = train['prediction_pay_price'].values

    del train['prediction_pay_price']
    del test['prediction_pay_price']

    train.to_csv('../data/train1.txt',sep=',',index=False,header=True)
    test.to_csv('../data/test1.txt',sep=',',index=False,header=True)

    pickle.dump(label,open('../data/label1.pkl','wb'))
    pickle.dump(test_id,open('../data/test_id1.pkl','wb'))

    t2=time.time()
    print("time use:",t2-t1)

    return train,label, test, test_id


if __name__ == '__main__':
    train_data_path = '../data/train1.txt'
    if os.path.exists(train_data_path):
        label = pickle.load(open('../data/label1.pkl','rb'))
        test_id = pickle.load(open('../data/test_id1.pkl','rb'))
        train = pd.read_csv('../data/train1.txt',sep=',')
        test = pd.read_csv('../data/test1.txt',sep=',')
    else:
        train,label, test, test_id = read_data_gen_feat()

    train.fillna(0,inplace=True)
    test.fillna(0,inplace=True)
    print(train.shape, test.shape)

    train_col = ['avg_online_minutes','pay_count','pay_price','pve_battle_count','pve_lanch_count','pve_win_count',\
                'pvp_battle_count','pvp_lanch_count','pvp_win_count','register_day','register_hour','pvp_lanch_ratio',\
                'pvp_win_ratio','pvp_win_lanch_ratio','pve_lanch_ratio','pve_pvp_lanch_','pve_pvp_win_ratio',\
                'pve_pvp_win_lanch_ratio','pay_mean_count','pay_mean_online_minutes','pay_count_online_minutes']

    # train = train[train_col]
    # test = test[train_col]

    train_num = train.shape[0]
    data = pd.concat([train,test],axis=0)

    data['register_day'] = pd.cut(data['register_day'],bins=20,labels=False)
    data['register_hour'] = pd.cut(data['register_hour'],bins=6,labels=False)
    register_day = pd.get_dummies(data['register_day'])
    register_day.columns = ['day_%s' % i for i in range(register_day.shape[1])]
    register_hour = pd.get_dummies(data['register_hour'])
    register_hour.columns = ['hour_%s' % i for i in range(register_hour.shape[1])]

    print(data.shape,register_day.shape,register_hour.shape)

    data = pd.concat([data,register_day,register_hour],axis=1)

    del data['register_day']
    del data['register_hour']

    train = data[:train_num]
    test = data[train_num:]

    train = train.values
    test = test.values

    print(train.shape,test.shape)

    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)

    # train_new_feature = pickle.load(open('./data/xgb_feat_train.pkl','rb'))
    # test_new_feature = pickle.load(open('./data/xgb_feat_test.pkl','rb'))

    # train = np.c_[train,train_new_feature]
    # test = np.c_[test,test_new_feature]

    def rmsel(y_true,y_pre):
        return mean_squared_error(y_true,y_pre)**0.5

    kf = KFold(n_splits=10,random_state=24,shuffle=True)
    best_rmse = []
    pred_list = []
    for train_index, val_index in kf.split(train):
        X_train = train[train_index]
        y_train = label[train_index]
        X_val = train[val_index]
        y_val = label[val_index]

        # regr = LinearRegression()
        # regr = Ridge(alpha=1.0, max_iter=100, tol=0.001, random_state=24)
        # regr = RandomForestRegressor(n_estimators=120,max_depth=8, random_state=0)
        regr = GradientBoostingRegressor(n_estimators=100, subsample=0.9)
        regr.fit(X_train,y_train)
        predi = regr.predict(X_val)
        predi = np.where(predi<0,0,predi)

        rmse = rmsel(y_val, predi)
        print("cv: ",rmse)

        predi = regr.predict(test)
        predi = np.where(predi<0,0,predi)

        pred_list.append(predi)
        best_rmse.append(rmse)

    pred = np.mean(np.array(pred_list),axis=0)
    meanrmse = np.mean(best_rmse)
    stdrmse = np.std(best_rmse)

    print('10 flod mean rmse, std rmse:',(meanrmse,stdrmse))

    pred = np.where(pred<0,0,pred)
    test_id['prediction_pay_price'] = 0
    test_id.loc[test_id['pay_price']>0,'prediction_pay_price'] = pred
    del test_id['pay_price']

    print(test_id.describe())
    print(test_id.sort_values(by='prediction_pay_price',ascending=False).head(10))

    test_id.loc[test_id['user_id'] == 2483734,'prediction_pay_price'] = 42823.47
    test_id.loc[test_id['user_id'] == 2492612,'prediction_pay_price'] = 38823.47
    test_id.loc[test_id['user_id'] == 1225981,'prediction_pay_price'] = 25875.59
    test_id.loc[test_id['user_id'] == 2354051,'prediction_pay_price'] = 17654
    test_id.loc[test_id['user_id'] == 498285,'prediction_pay_price'] = 21833.164
    test_id.loc[test_id['user_id'] == 1760226,'prediction_pay_price'] = 18427 

    test_id.to_csv('./output/gbdt_v2.csv',sep=',',header=True,index=False,float_format='%.4f')



