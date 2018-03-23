"""
    文件配置
"""
# 数据文件配置
test_file_name = "data/train_data/test.csv"
train_file_name = "data/train_data/train.csv"
# 模型 performance 文件配置
performance_file_name = "src/log/model_performance.txt"
performance_file_mode = "a+"
# 结果文件配置
submission_file_name = "data/submission/submission.csv"
# 参数存储文件配置
params_file_mode = "w"
tree_params_file_name = "src/params/model.txt"
params_file_name = "src/params/params.txt"

# 分隔符
csv_seperator = ","
"""
    GridSearch 中的参数调整
"""
n_threads = 2  # CPU cores
metric = ['binary_error', 'auc']

# 通过学习固定的参数
cv_learning_rates = [0.05]
cv_boosting_type = ['gbdt']
cv_n_estimators = [100]
cv_feature_fraction = [0.9]

# 用于防止模型过拟合的参数
# 通过学习后的固定参数
cv_num_leaves = [10]
cv_lambda_l1 = [0.9]
cv_lambda_l2 = [1]
cv_bagging_fraction = [0.9]
cv_bagging_freq = [0]
"""
    最后得到的模型参数
"""
cv_split_size = 0.07
boosting_type = 'gbdt'
n_estimators = 100
feature_fraction = 0.9
learning_rate = 0.05
early_stopping_rounds = 50
verbose_eval = 50
num_leaves = 10
lambda_l1 = 0.9
lambda_l2 = 1
bagging_fraction = 0.9
bagging_freq = 0

"""
    项目中某些功能的控制开关
"""

can_plot_tree = False
can_show_metric = True
can_grid_search_hyper_params = True
can_save_best_params = True
