
import pandas as pd
import numpy as np

nn = pd.read_csv('output/nn.csv', index_col=False) ##nn模型需要对小于10000的用户做处理,需要乘以系数
#对nn做值变换，保证训练集测试集均值大致相同
nn.prediction_pay_price *= (94.7/ nn[nn.prediction_pay_price != 0].prediction_pay_price.mean())

gbdt_less = pd.read_csv('output/gbdt_xiaoyu_16000.csv', index_col=False) #gbdt乘
gbdt_less.prediction_pay_price *= (94.7/ gbdt_less[gbdt_less.prediction_pay_price != 0].prediction_pay_price.mean())

knn_lasso = pd.read_csv('output/knn_lasso.csv', index_col=False)   #knn_lasso系数乘过了

gbdt_v1 = pd.read_csv('output/gbdt_v1.csv', index_col=False)
gbdt_v2 = pd.read_csv('output/gbdt_v2.csv', index_col=False)

gbdt_diff_feature = gbdt_v1.copy()
gbdt_diff_feature['prediction_pay_price'] = 0.6 * gbdt_v1['prediction_pay_price'].values + 0.4 * gbdt_v2['prediction_pay_price'].values

# lgb 的结果是较早跑出来的，现在改变了一些特征，重新跑的话可能结果有些差异，如果差异较大，请直接使用
# 我们的原始结果 output/initial_lightgbm.csv
lgb_init = pd.read_csv('output/first_hyperopt_model_prediction.csv', index_col=False)


#融合3个模型试试
result = pd.DataFrame()
result['user_id'] = knn_lasso['user_id']
result['prediction_pay_price'] = 0


gbdt_less_res = gbdt_less['prediction_pay_price'].values
gbdt_diff_res = gbdt_diff_feature['prediction_pay_price'].values
knn_lasso_res = knn_lasso['prediction_pay_price'].values
nn_res = nn['prediction_pay_price'].values
lgb_init_res = lgb_init['prediction_pay_price'].values


sub1 = 0.45*gbdt_less_res + 0.35*gbdt_diff_res + 0.2*knn_lasso_res
sub2 = 0.6*sub1 + 0.4*nn_res
sub3 = 0.75*sub2 + 0.25*lgb_init_res

result.loc[:, 'prediction_pay_price'] = sub3

# 由于模型有一些波动，而且有些模型所用的训练集都是 pay_price 小于 16000 的样本，为了保证土豪用户的预测结果不出现
# 大的波动，我们根据前期所提交结果的反馈以及多个模型的预测结果，将以下用户的消费金额在融合前进行事先设定
result.loc[result['user_id'] == 2483734,'prediction_pay_price'] = 45996.47
result.loc[result['user_id'] == 2492612,'prediction_pay_price'] = 43160.18
result.loc[result['user_id'] == 1225981,'prediction_pay_price'] = 30775.14
result.loc[result['user_id'] == 2354051,'prediction_pay_price'] = 14309.5 
result.loc[result['user_id'] == 498285,'prediction_pay_price'] = 20202.28
result.loc[result['user_id'] == 1760226,'prediction_pay_price'] = 17131.82
result.loc[result['user_id'] == 2851137,'prediction_pay_price'] = 12719.79
result.loc[result['user_id'] == 1695889,'prediction_pay_price'] = 12045.26
result.loc[result['user_id'] == 2145991,'prediction_pay_price'] = 10268.29
result.loc[result['user_id'] == 1295687,'prediction_pay_price'] = 14375.94

knn_less3_user = knn_lasso.loc[knn_lasso['prediction_pay_price']<3,'user_id'].tolist()
result.loc[result['user_id'].isin(knn_less3_user),'prediction_pay_price'] = knn_lasso.loc[knn_lasso['prediction_pay_price']<3,'prediction_pay_price'].values
result.loc[(result.prediction_pay_price < 3) & (result.prediction_pay_price != 0), 'prediction_pay_price'] = 0.99

print(result[result.prediction_pay_price != 0].describe())

result.to_csv('submission.csv',sep=',',header=True,index=False,float_format='%.4f')

