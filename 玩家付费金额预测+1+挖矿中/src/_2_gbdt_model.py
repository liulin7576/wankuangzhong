
import gc
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, ElasticNetCV, RidgeCV

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
    #取pay_price不为0的用户
    data = data[data.pay_price != 0]

    return data,test_usid

def get_poly_fea(data):
        fea_list = ['ivory_add_value', 'wood_add_value', 'stone_add_value', 'general_acceleration_add_value', 'ivory_reduce_value', 'meat_add_value', 'wood_reduce_value', 
                'training_acceleration_add_value']
        for i in fea_list:
            data[i + str(i) + 'sqrt'] = data[i] ** 0.5
        return data

def process(data):
    data['register_hour'] = data['register_time'].map(lambda x : int(x[11:13]))
    data['register_time_day'] = data['register_time'].map(lambda x : x[5:10])

    data.loc[:,'is_pay_price45'] = data['pay_price'].map(lambda x: 1 if x>0 else 0)
    data.loc[:,'is_pay_099'] = data['pay_price'].map(lambda x: 1 if x<1 else 0)

    have_pay_price_mean = data.groupby(['register_time_day'])['is_pay_price45'].mean()
    have_pay_099_mean = data.groupby(['register_time_day'])['is_pay_099'].mean()
    pay_099_ration = data.loc[data['is_pay_price45']>0,:].copy()
    pay_099_ration = pay_099_ration.groupby(['register_time_day'])['is_pay_099'].mean()
    data['have_pay_price_mean_hour'] = data['register_hour'].map(lambda x : have_pay_price_mean[x])
    data['have_pay_099_mean_hour'] = data['register_hour'].map(lambda x : have_pay_099_mean[x])
    data['pay_099_ration_hour'] = data['register_hour'].map(lambda x : pay_099_ration[x])
    del data['register_hour']
    del data['register_time_day']
    
    import time
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= (3600 * 24)  #这里试试天数会不会更有效果
    data['regedit_diff_day'] = (a - min(a))
    
     #这个特征是顾客在3月双休日点击APP的总次数
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

    #这个特征是顾客在3月双休日点击APP的总次数
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-03-10', '2018-02-19']
    new['date_holiday'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_holiday', 'user_id']], on='user_id',how='left')
    
    #这个特征是顾客在3月双休日点击APP的总次数
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
    data['month'] = data['register_time'].apply(lambda x:x.month)
    data['hour'] = data['register_time'].apply(lambda x:x.hour)
    data['minute'] = data['register_time'].apply(lambda x:x.hour*60 + x.minute)
    for i in ['dow', 'doy', 'month']:
        a = pd.get_dummies(data[i], prefix = i)
        data = pd.concat([data, a], axis = 1)
        del data[i]

    data = get_poly_fea(data)

    del data['register_time']
    
    data.loc[:,'wood_reduce_ratio'] = data['wood_reduce_value'] / (data['wood_add_value']+1e-4)
    data.loc[:,'stone_reduce_ratio'] = data['stone_reduce_value'] / (data['stone_add_value']+1e-4)
    data.loc[:,'ivory_reduce_ratio'] = data['ivory_reduce_value'] / (data['ivory_add_value']+1e-4)
    data.loc[:,'meat_reduce_ratio'] = data['meat_reduce_value'] / (data['meat_add_value']+1e-4)
    data.loc[:,'magic_reduce_ratio'] = data['magic_reduce_value'] / (data['magic_add_value']+1e-4)
    data.loc[:,'infantry_reduce_ratio'] = data['infantry_reduce_value'] / (data['infantry_add_value']+1e-4)
    data.loc[:,'cavalry_reduce_ratio'] = data['cavalry_reduce_value'] / (data['cavalry_add_value']+1e-4)
    data.loc[:,'shaman_reduce_ratio'] = data['shaman_reduce_value'] / (data['shaman_add_value']+1e-4)
    data.loc[:,'wound_infantry_reduce_ratio'] = data['wound_infantry_reduce_value'] / (data['wound_infantry_add_value']+1e-4)
    data.loc[:,'wound_cavalry_reduce_ratio'] = data['wound_cavalry_reduce_value'] / (data['wound_cavalry_add_value']+1e-4)
    data.loc[:,'wound_shaman_reduce_ratio'] = data['wound_shaman_reduce_value'] / (data['wound_shaman_add_value']+1e-4)
    data.loc[:,'general_acceleration_reduce_ratio'] = data['general_acceleration_reduce_value'] / (data['general_acceleration_add_value']+1e-4)
    data.loc[:,'building_acceleration_reduce_ratio'] = data['building_acceleration_reduce_value'] / (data['building_acceleration_add_value']+1e-4)
    data.loc[:,'reaserch_acceleration_reduce_ratio'] = data['reaserch_acceleration_reduce_value'] / (data['reaserch_acceleration_add_value']+1e-4)
    data.loc[:,'training_acceleration_reduce_ratio'] = data['training_acceleration_reduce_value'] / (data['training_acceleration_add_value']+1e-4)
    data.loc[:,'treatment_acceleraion_reduce_ratio'] = data['treatment_acceleration_reduce_value'] / (data['treatment_acceleraion_add_value']+1e-4)
    data.loc[:,'wood_add_sub_reduce'] = np.abs(data['wood_add_value'] - data['wood_reduce_value'])
    data.loc[:,'stone_add_sub_reduce'] = np.abs(data['stone_add_value'] - data['stone_reduce_value'])
    data.loc[:,'ivory_add_sub_reduce'] = np.abs(data['ivory_add_value'] - data['ivory_reduce_value'])
    data.loc[:,'meat_add_sub_reduce'] = np.abs(data['meat_add_value'] - data['meat_reduce_value'])
    data.loc[:,'magic_add_sub_reduce'] = np.abs(data['magic_add_value'] - data['magic_reduce_value'])
    data.loc[:,'infantry_add_sub_reduce'] = np.abs(data['infantry_add_value'] - data['infantry_reduce_value'])
    data.loc[:,'cavalry_add_sub_reduce'] = np.abs(data['cavalry_add_value'] - data['cavalry_reduce_value'])
    data.loc[:,'shaman_add_sub_reduce'] = np.abs(data['shaman_add_value'] - data['shaman_reduce_value'])
    data.loc[:,'wound_infantry_add_sub_reduce'] = np.abs(data['wound_infantry_add_value'] - data['wound_infantry_reduce_value'])
    data.loc[:,'wound_cavalry_add_sub_reduce'] = np.abs(data['wound_cavalry_add_value'] - data['wound_cavalry_reduce_value'])
    data.loc[:,'wound_shaman_add_sub_reduce'] = np.abs(data['wound_shaman_add_value'] - data['wound_shaman_reduce_value'])
    data.loc[:,'general_acceleration_add_sub_reduce'] = np.abs(data['general_acceleration_add_value'] - data['general_acceleration_reduce_value'])
    data.loc[:,'building_acceleration_add_sub_reduce'] = np.abs(data['building_acceleration_add_value'] - data['building_acceleration_reduce_value'])
    data.loc[:,'reaserch_acceleration_add_sub_reduce'] = np.abs(data['reaserch_acceleration_add_value'] - data['reaserch_acceleration_reduce_value'])
    data.loc[:,'training_acceleration_add_sub_reduce'] = np.abs(data['training_acceleration_add_value'] - data['training_acceleration_reduce_value'])
    data.loc[:,'treatment_acceleration_add_sub_reduce'] = np.abs(data['treatment_acceleraion_add_value'] - data['treatment_acceleration_reduce_value'])
    log_col = ['wood_add_value','wood_reduce_value','stone_add_value','stone_reduce_value','ivory_add_value',
                'ivory_reduce_value','meat_add_value','meat_reduce_value','magic_add_value','magic_reduce_value',
                'infantry_add_value','infantry_reduce_value','cavalry_add_value','cavalry_reduce_value','shaman_add_value',
                'shaman_reduce_value','wound_infantry_add_value','wound_infantry_reduce_value','wound_cavalry_add_value',
                'wound_cavalry_reduce_value','wound_shaman_add_value','wound_shaman_reduce_value',
                'general_acceleration_add_value','general_acceleration_reduce_value','building_acceleration_add_value',
                'building_acceleration_reduce_value','reaserch_acceleration_add_value','reaserch_acceleration_reduce_value',
                'training_acceleration_add_value','training_acceleration_reduce_value','treatment_acceleraion_add_value',
                'treatment_acceleration_reduce_value']
    for col in log_col:
        data[col] = data[col].map(lambda x : np.log1p(x))
    # 物资消耗统计
    ratio_col = ['wood_reduce_ratio','stone_reduce_ratio','ivory_reduce_ratio','meat_reduce_ratio','magic_reduce_ratio',\
                'infantry_reduce_ratio','cavalry_reduce_ratio','shaman_reduce_ratio','wound_infantry_reduce_ratio',\
                'wound_cavalry_reduce_ratio','wound_shaman_reduce_ratio','general_acceleration_reduce_ratio',\
                'building_acceleration_reduce_ratio','reaserch_acceleration_reduce_ratio','training_acceleration_reduce_ratio',\
                'treatment_acceleraion_reduce_ratio']
    data.loc[:,'ratio_max'] = data[ratio_col].max(1)
    data.loc[:,'ratio_min'] = data[ratio_col].min(1)
    data.loc[:,'ratio_mean'] = data[ratio_col].mean(1)
    data.loc[:,'ratio_sum'] = data[ratio_col].sum(1)
    data.loc[:,'ratio_std'] = data[ratio_col].std(1)
    data.loc[:,'ratio_median'] = data[ratio_col].median(1)
    # data.loc[:,'ratio_mode'] = data[ratio_col].mode(1)

    # 物资生产统计
    add_col = ['wood_add_value','stone_add_value','ivory_add_value','meat_add_value','magic_add_value',\
                'infantry_add_value','cavalry_add_value','shaman_add_value','wound_infantry_add_value',\
                'wound_cavalry_add_value','wound_shaman_add_value','general_acceleration_add_value',\
                'building_acceleration_add_value','reaserch_acceleration_add_value','training_acceleration_add_value',\
                'treatment_acceleraion_add_value']
    data.loc[:,'add_max'] = data[add_col].max(1)
    data.loc[:,'add_min'] = data[add_col].min(1)
    data.loc[:,'add_mean'] = data[add_col].mean(1)
    data.loc[:,'add_sum'] = data[add_col].sum(1)
    data.loc[:,'add_std'] = data[add_col].std(1)
    data.loc[:,'add_median'] = data[add_col].median(1)
    # data.loc[:,'add_mode'] = data[add_col].mode(1)
    
    reduce_col = ['wood_reduce_value','stone_reduce_value','ivory_reduce_value','meat_reduce_value','magic_reduce_value',\
                'infantry_reduce_value','cavalry_reduce_value','shaman_reduce_value','wound_infantry_reduce_value',\
                'wound_cavalry_reduce_value','wound_shaman_reduce_value','general_acceleration_add_value',\
                'building_acceleration_reduce_value','reaserch_acceleration_reduce_value','training_acceleration_reduce_value',\
                'treatment_acceleration_reduce_value']
    data.loc[:,'reduce_max'] = data[reduce_col].max(1)
    data.loc[:,'reduce_min'] = data[reduce_col].min(1)
    data.loc[:,'reduce_mean'] = data[reduce_col].mean(1)
    data.loc[:,'reduce_sum'] = data[reduce_col].sum(1)
    data.loc[:,'reduce_std'] = data[reduce_col].std(1)
    data.loc[:,'reduce_median'] = data[reduce_col].median(1)
    # data.loc[:,'reduce_mode'] = data[reduce_col].mode(1)


    # 建筑等级统计
    bd_col = ['bd_training_hut_level','bd_healing_lodge_level','bd_stronghold_level','bd_outpost_portal_level',
            'bd_barrack_level','bd_healing_spring_level','bd_dolmen_level','bd_guest_cavern_level','bd_warehouse_level',
            'bd_watchtower_level','bd_magic_coin_tree_level','bd_hall_of_war_level','bd_market_level','bd_hero_gacha_level',
            'bd_hero_strengthen_level','bd_hero_pve_level']
    data.loc[:,'bd_max'] = data[bd_col].max(1)
    data.loc[:,'bd_min'] = data[bd_col].min(1)
    data.loc[:,'bd_mean'] = data[bd_col].mean(1)
    data.loc[:,'bd_sum'] = data[bd_col].sum(1)
    data.loc[:,'bd_std'] = data[bd_col].std(1)
    data.loc[:,'bd_median'] = data[bd_col].median(1)
    # data.loc[:,'bd_mode'] = data[bd_col].mode(1)
    
    # 科研 tier 统计
    tier_col = ['sr_infantry_tier_2_level','sr_cavalry_tier_2_level','sr_shaman_tier_2_level',
                'sr_infantry_tier_3_level','sr_cavalry_tier_3_level','sr_shaman_tier_3_level',
                'sr_infantry_tier_4_level','sr_cavalry_tier_4_level','sr_shaman_tier_4_level']

    data.loc[:,'infantry_tier_sum'] = data[[tier_col[0],tier_col[3],tier_col[6]]].sum(1)
    data.loc[:,'cavalry_tier_sum'] = data[[tier_col[1],tier_col[4],tier_col[7]]].sum(1)
    data.loc[:,'shaman_tier_sum'] = data[[tier_col[2],tier_col[5],tier_col[8]]].sum(1)

    data.loc[:,'sr_tier_sum'] = data[tier_col].sum(1)
    data.loc[:,'sr_tier_max'] = data[tier_col].max(1)
    data.loc[:,'sr_tier_min'] = data[tier_col].min(1)
    data.loc[:,'sr_tier_mean'] = data[tier_col].mean(1)
    data.loc[:,'sr_tier_std'] = data[tier_col].std(1)
    data.loc[:,'sr_tier_median'] = data[tier_col].median(1)


    # 攻击
    atk_col = ['sr_infantry_atk_level','sr_cavalry_atk_level','sr_shaman_atk_level','sr_troop_attack_level']
    data.loc[:,'atk_sum'] = data[atk_col].sum(1)
    data.loc[:,'atk_mean'] = data[atk_col].mean(1)
    data.loc[:,'atk_max'] = data[atk_col].max(1)
    data.loc[:,'atk_std'] = data[atk_col].std(1)
    data.loc[:,'atk_min'] = data[atk_col].min(1)
    data.loc[:,'atk_median'] = data[atk_col].median(1)
    # data.loc[:,'atk_mode'] = data[atk_col].mode(1)
    
     # 防御
    def_col = ['sr_infantry_def_level','sr_cavalry_def_level','sr_shaman_def_level','sr_troop_defense_level']
    data.loc[:,'def_sum'] = data[def_col].sum(1)
    data.loc[:,'def_mean'] = data[def_col].mean(1)
    data.loc[:,'def_max'] = data[def_col].max(1)
    data.loc[:,'def_min'] = data[def_col].min(1)
    data.loc[:,'def_std'] = data[def_col].std(1)
    # data.loc[:,'def_mode'] = data[def_col].mode(1)

    # 生命力
    hp_col = ['sr_infantry_hp_level','sr_cavalry_hp_level','sr_shaman_hp_level']
    data.loc[:,'hp_sum'] = data[hp_col].sum(1)
    data.loc[:,'hp_mean'] = data[hp_col].mean(1)
    data.loc[:,'hp_max'] = data[hp_col].max(1)
    # data.loc[:,'hp_mode'] = data[hp_col].mode(1)
    
    # 各种 level 统计
    level_col = ['sr_construction_speed_level','sr_hide_storage_level','sr_troop_consumption_level',
                'sr_rss_a_prod_levell','sr_rss_b_prod_level','sr_rss_c_prod_level',
                'sr_rss_d_prod_level','sr_rss_a_gather_level','sr_rss_b_gather_level',
                'sr_rss_c_gather_level','sr_rss_d_gather_level','sr_troop_load_level','sr_rss_e_gather_level',
                'sr_rss_e_prod_level','sr_outpost_durability_level','sr_outpost_tier_2_level',
                'sr_healing_space_level','sr_gathering_hunter_buff_level','sr_healing_speed_level',
                'sr_outpost_tier_3_level','sr_alliance_march_speed_level','sr_pvp_march_speed_level',
                'sr_gathering_march_speed_level','sr_outpost_tier_4_level','sr_guest_troop_capacity_level',
                'sr_march_size_level','sr_rss_help_bonus_level',]
    data.loc[:,'same_level_sum'] = data[level_col].sum(1)
    data.loc[:,'same_level_mean'] = data[level_col].mean(1)
    data.loc[:,'same_level_max'] = data[level_col].max(1)
    data.loc[:,'same_level_std'] = data[level_col].std(1)
    data.loc[:,'same_level_min'] = data[level_col].min(1)
    data.loc[:,'same_level_median'] = data[level_col].median(1)
    # data.loc[:,'same_level_mode'] = data[level_col].mode(1)
    
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

    data.loc[:,'time_lanch_per'] = data['avg_online_minutes'] / (data['pve_pvp_lanch_count'] + 1e-4)
    data.loc[:,'money_lanch_per'] = data['pay_price'] / (data['pve_pvp_lanch_count'] + 1e-4)
    
    data.loc[:,'time_battle_per'] = data['avg_online_minutes'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'money_battle_per'] = data['pay_price'] / (data['pve_pvp_battle_count'] + 1e-4)

    # data.loc[:,'mean_pay_price'] = data['pay_price'] / 7

    data.loc[:,'time_win_per'] = data['avg_online_minutes'] / (data['pve_pvp_win_count'] + 1e-4)
    data.loc[:,'money_win_per'] = data['pay_price'] / (data['pve_pvp_win_count'] + 1e-4)
    data.loc[:,'pay_count_win_per'] = data['pay_count'] / (data['pve_pvp_win_count'] + 1e-4)

    return data


def split(data):
    train = data[data.prediction_pay_price != -1]
    test = data[data.prediction_pay_price == -1]
    test_usid = test.user_id
    del test['user_id']
    del test['prediction_pay_price']
    test_X = test.values.astype(np.float32)

    del train['user_id']
    train = train.loc[train['prediction_pay_price']<16000,:]
    y = train['prediction_pay_price'].values.astype(np.float32)
    
    del train['prediction_pay_price']
    X = train.values.astype(np.float32)
    col = train.columns

    return X, y, test_X, test_usid


def select_feat(X, y, test_X):
    #用模型选择特征
    xgb_regressor = xgb.XGBRegressor()
    model_lasso = LassoCV(alphas = [1, 0.1, 0.005,0.003,  0.001, 0.0005, 0.0001])
    sfm = SelectFromModel(xgb_regressor)
    sfm.fit(X, y)
    X = sfm.transform(X)
    test_X = sfm.transform(test_X)

    return X, test_X

def rmsel(y_true,y_pre):
    return mean_squared_error(y_true,y_pre)**0.5


if __name__ == '__main__':

    train_data_path = '../data/gbdt2_train_data.pkl'
    if os.path.exists(train_data_path):
        X, y,test_usid, test_X, test_with_pay_usid = pickle.load(open(train_data_path,'rb'))
    else:
        data, test_usid = read_data()
        data = process(data)

        X, y, test_X, test_with_pay_usid  = split(data)
        X, test_X = select_feat(X, y, test_X)

        pickle.dump((X, y,test_usid ,test_X, test_with_pay_usid),open(train_data_path,'wb'))

    print("X,test_X :",X.shape,test_X.shape)

    #用10折交叉验证预测结果
    kf = KFold(n_splits=10,random_state=24,shuffle=True)
    best_rmse = []
    pred_list = []
    for train_index, val_index in kf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        # regr = LinearRegression()
        # regr = Ridge(alpha=1.0, max_iter=100, tol=0.001, random_state=24)
        # regr = RandomForestRegressor(n_estimators=120,max_depth=8, random_state=0)

        regr = GradientBoostingRegressor(n_estimators=100, subsample=0.9)
        regr.fit(X_train,y_train)
        predi = regr.predict(X_val)
        predi = np.where(predi<0,0,predi)

        rmse = rmsel(y_val, predi)
        print("cv: ",rmse)

        predi = regr.predict(test_X)
        predi = np.where(predi<0,0,predi)

        pred_list.append(predi)
        best_rmse.append(rmse)

    pred = np.mean(np.array(pred_list),axis=0)
    meanrmse = np.mean(best_rmse)
    stdrmse = np.std(best_rmse)
    print('10 flod mean rmse, std rmse:',(meanrmse,stdrmse))


    test_with_pay = pd.DataFrame()
    test_with_pay['user_id'] = test_with_pay_usid
    pred[pred < 1] = 0.99
    test_with_pay['prediction_pay_price'] = pred
    test_with_pay.describe()

    sub = pd.DataFrame()
    sub['user_id'] = test_usid.values
    sub['prediction_pay_price'] = 0
    sub.loc[sub.user_id.isin(test_with_pay.user_id), 'prediction_pay_price'] = \
                test_with_pay['prediction_pay_price'].values * 1.432
    print(sub.head(), '\n')
    print(sub.describe())

    sub.to_csv('./output/gbdt_xiaoyu_16000.csv', index=False)



	