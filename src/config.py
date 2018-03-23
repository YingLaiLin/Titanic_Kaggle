# 数据存储目录
test_file_name = "data/train_data/test.csv"
train_file_name = "data/train_data/train.csv"
# 日志目录
log_file_name = "src/log/score_record.log"
# 结果文件目录
submission_file_name = "data/submission/submission.csv"
# 参数存储目录
params_file_name = "src/params/model.txt"
# 分隔符
csv_seperator = ","
# GridSearch 中的参数调整
split_size = 0.07
learning_rates = [0.01]
boosting_type = ['dart', 'gbdt']
reg_lambda = [1, 1.2, 1.4]
reg_alpha = [1, 1.2]
n_estimators = [10, 12, 14, 16]
metric = 'binary_error'
early_stopping_rounds = 300
verbose_eval = 50
