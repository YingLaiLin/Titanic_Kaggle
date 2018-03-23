# -*- coding: utf-8 -*-
import logging as log
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src import config


# TODO 引入 stacking
# TODO 引入交叉验证选择最佳模型
# TODO 合并 map_feature 和 select_feature 两个函数

def main():
    config_log()
    train_data, labeled_size, labels, Ids = pre_process_data()
    bestParams = get_offline_result(train_data, labeled_size, labels)
    model = train_model(bestParams, train_data, labeled_size, labels)
    evaluate_and_save(model, train_data, labeled_size, Ids)
    sys.exit(0)


def config_log(log_filename=config.log_file_name):
    """
        配置输出格式
    :param log_filename: log 日志文件的存放地址
    :return:
    """
    log.basicConfig(level=log.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %('
                           'levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=log_filename,
                    filemode='a+')


def pre_process_data(train_filename=config.train_file_name,
                     test_filename=config.test_file_name):
    """
        将测试数据和训练数据共同进行缺失值、特征选择处理
    :param train_filename:
    :param test_filename:
    :return: 返回整合后的数据以及训练数据的大小
    """

    train_data = load_data(train_filename)
    test_data = load_data(test_filename)
    integrated_data = pd.concat([train_data, test_data])
    integrated_data = handle_na(integrated_data)
    integrated_data, labels, Ids = extract_features(integrated_data)
    integrated_data = map_feature(integrated_data)
    return integrated_data, len(train_data), labels, Ids


def load_data(filename, sep=config.csv_seperator):
    return pd.read_csv(filename, sep=sep)


def get_offline_result(train_data, labeled_size, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        train_data[:labeled_size], labels[:labeled_size],
        test_size=config.cv_split_size, random_state=42)
    gridParams = {
        'learning_rate': config.cv_learning_rates,
        'n_estimators': config.cv_n_estimators,
        'num_leaves': config.cv_num_leaves,
        'boosting_type': config.cv_boosting_type, 'objective': ['binary'],
        'lambda_l1': config.cv_lambda_l1, 'lambda_l2': config.cv_lambda_l2,
        'random_state': [501], 'feature_fraction': config.cv_feature_fraction,
        # 'colsample_bytree': [0.64, 0.65, 0.66], 'subsample': [0.7, 0.75],
        # 'reg_alpha': [1, 1.2], 'reg_lambda': [1, 1.2, 1.4],
        }
    clf = lgb.LGBMRegressor(boosting_type='dart', objective='binary')
    grid = GridSearchCV(clf, gridParams, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid


def train_model(grid, train, labeled_size, labels):
    params = {
        'boosting_type': config.boosting_type, 'objective': 'binary',
        'metric': config.metric, 'learning_rate': config.learning_rate,
        # 根据 GridSearchCV 中的参数进行设置
        'nthread': config.n_threads, 'n_estimators': config.n_estimators,
        'feature_fraction': 0.9, 'random_state': 501,
        'lambda_l1': config.lambda_l1, 'lambda_l2': config.lambda_l2,
        'max_depth': -1, 'num_leaves': config.num_leaves,
        }
    x_train, x_test, y_train, y_test = train_test_split(train[:labeled_size],
                                                        labels[:labeled_size],
                                                        test_size=config.cv_split_size,
                                                        random_state=42)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    eval_results = {}
    log.info('\ngrid best params {}'.format(grid.best_params_))
    gbm = lgb.train(params, train_set=train_data, num_boost_round=2000,
                    valid_sets=[train_data, test_data],
                    evals_result=eval_results, verbose_eval=50,
                    early_stopping_rounds=config.early_stopping_rounds)
    log.info("best iterations {}".format(gbm.best_iteration))
    predicted = np.round(gbm.predict(x_test))
    record_performance(y_test, predicted)
    show_model_performance(gbm, eval_results)
    return gbm


def evaluate_and_save(clf, train, labeled_size, Ids):
    # 存储模型参数
    clf.save_model(config.params_file_name)
    test = train[labeled_size:]
    res = list(map(lambda x: 1 if x == 1.0 else 0, np.round(
        clf.predict(test, num_iteration=clf.best_iteration))))

    pd.DataFrame({
        'PassengerId': Ids[labeled_size:], 'Survived': res
        }).to_csv(config.submission_file_name, index=False)


def extract_features(train_data):
    # 提取是否有家人这一特征
    train_data['Alone'] = (train_data['SibSp'] == 0) & (
            train_data['Parch'] == 0)
    # 提取名字作为特征
    train_data['Title'] = train_data.Name.str.extract(r'([A-Za-z]+)\.',
                                                      expand=False)
    train_data['Title'] = train_data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
         'Jonkheer', 'Dona'], 'Rare')

    train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
    train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
    train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_data.Title = train_data.Title.map(title_mapping)
    # 缺失姓名的值默认为 0
    train_data.Title = train_data.Title.fillna(0)
    labels = train_data.Survived
    Ids = train_data.PassengerId
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare", "Alone",
                        "Title"]

    train = train_data[selected_columns]
    return train, labels, Ids


def map_feature(train_data):
    columns = train_data.columns
    """
     因为 Fare,Age 是一些连续值, 难以用于预测, 
     故将按照4分位数分为4个区间.[0,.25,.5,.75,1.]
    """
    if "Fare" in columns:
        train_data.Fare = pd.qcut(train_data.Fare, q=4, labels=False)
    if "Age" in columns:
        train_data.Age = pd.qcut(train_data.Age, q=4, labels=False)  #

    # 直接利用哑变量方法来对变量进行处理, 也可以通过 map 来进行变换
    data_dum = pd.get_dummies(train_data, drop_first=True)
    return data_dum


def handle_na(train_data):
    # 年龄 和 费用用平均值填充
    train_data.Age = train_data.Age.fillna(train_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(train_data.Fare.median())
    # 上船港口默认为 S
    train_data.Embarked = train_data.Embarked.fillna('S')
    return train_data


def show_model_performance(gbm, evals_result):
    # show model importance
    # lgb.plot_importance(gbm)
    # Show Decision Tree
    if config.is_plot_tree:
        graph = lgb.create_tree_digraph(gbm, name='Tree84')
        graph.render(view=True)
    if config.is_show_metric:
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        for index in range(len(config.metric)):
            lgb.plot_metric(evals_result, config.metric[index],
                            title=config.metric[index], ax=axs[index])
    plt.show()


def record_performance(y_test, predicted):
    log.info('score is: {}\n'.format(accuracy_score(y_test, predicted)))
    log.info(' \n {}'.format(classification_report(y_test, predicted,
                                                   target_names=['Died',
                                                                 'Survived'])))
    log.info("------------------------------\n")


if __name__ == "__main__":
    main()
