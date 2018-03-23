# 数据文件配置
test_file_name = "data/train_data/test.csv"
train_file_name = "data/train_data/train.csv"
# 日志文件配置
log_file_name = "src/log/model_performance.log"
# 结果文件配置
submission_file_name = "data/submission/submission.csv"
# 参数存储文件配置
params_file_name = "src/params/model.txt"
# 分隔符
csv_seperator = ","
# GridSearch 中的参数调整
n_threads = 2  # CPU cores
metric = ['binary_error', 'auc']

# cv_learning_rates = [0.01, 0.05, 0.001, 0.005]
# cv_boosting_type = ['dart', 'gbdt', 'goss']
# cv_feature_fraction = [0.7, 0.8, 0.9, 1]
# cv_n_estimators = [i * 50 for i in range(1, 10)]
# 通过学习固定参数
cv_learning_rates = [0.05]
cv_boosting_type = ['gbdt']
cv_n_estimators = [100]
cv_feature_fraction = [0.9]

# 用于防止模型过拟合的参数
# cv_lambda_l1 = [0.9, 1.2, 1, 1.5]
# cv_lambda_l2 = [0.9, 1.2, 1, 1.5]
# 通过学习固定参数
cv_num_leaves = [10]
cv_lambda_l1 = [0.9]
cv_lambda_l2 = [1]
# 最后得到的模型参数
cv_split_size = 0.10
boosting_type = 'gbdt'
n_estimators = 100
feature_fraction = 0.9
learning_rate = 0.05
early_stopping_rounds = 50
verbose_eval = 50
num_leaves = 10
lambda_l1 = 0.9
lambda_l2 = 1
# 控制画图
is_plot_tree = False
is_show_metric = True
