# -*- coding: utf-8 -*-
import logging as log
import sys

import lightgbm as lgb
import matplotlib.gridspec as gridspec
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
# TODO 为 train_model 的过程添加可视化过程 **
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
        test_size=config.split_size, random_state=42)
    gridParams = {
        'learning_rate': config.learning_rates,
        'n_estimators': config.n_estimators, 'num_leaves': [6, 8, 12, 16],
        'boosting_type': config.boosting_type,
        'objective': ['binary'], 'random_state': [501],  # Updated from 'seed'
        # 'colsample_bytree': [0.64, 0.65, 0.66], 'subsample': [0.7, 0.75],
        # 'reg_alpha': [1, 1.2], 'reg_lambda': [1, 1.2, 1.4],
        }
    # clf = lgb.LGBMRegressor(boosting_type='dart', objective='binary',
    # nthread=5,
    #                         silent=True, max_depth=-1, max_bin=128,
    #                         subsample_for_bin=500, subsample=1,
    #                         subsample_freq=1, min_split_gain=0.5,
    #                         min_child_weight=1, min_child_samples=5,
    #                         scale_pos_weight=1)
    clf = lgb.LGBMRegressor()
    grid = GridSearchCV(clf, gridParams, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid


def train_model(grid, train, labeled_size, labels):
    params = {
        'boosting_type': 'dart', 'objective': 'binary', 'metric': config.metric,
        'learning_rate': 0.01, 'max_depth': -1, # 根据 GridSearchCV 中的参数进行设置
        'nthread': 5, 'n_estimators': 16, 'num_leaves': 16, 'random_state': 501,

        }
    x_train, x_test, y_train, y_test = train_test_split(train[:labeled_size],
                                                        labels[:labeled_size],
                                                        test_size=config.split_size,
                                                        random_state=42)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    eval_results = {}
    log.info('\ngrid best params {}'.format(grid.best_params_))
    gbm = lgb.train(params, train_set=train_data, num_boost_round=2000,
                    valid_sets=[train_data, test_data],
                    evals_result=eval_results, verbose_eval=50,
                    early_stopping_rounds=config.early_stopping_rounds, )
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


# @deprecated
def visualize_loss(clf, evaluations, loss_evaluation):
    fig = plt.figure(figsize=(60, 40))
    gs = gridspec.GridSpec(6, 6)
    # Plot the log loss and AUC during training
    ax_loss = plt.subplot(gs[0, :2])
    ax_loss.plot(evaluations['Train'][loss_evaluation[0]], label='Train')
    ax_loss.plot(evaluations['Test'][loss_evaluation[0]], label='Test')
    ax_loss.plot(evaluations['Train'][loss_evaluation[1]], label='Train')
    ax_loss.plot(evaluations['Test'][loss_evaluation[1]], label='Test')
    ax_loss.set_ylabel('Log loss')
    ax_loss.set_xlabel('Boosting round')
    ax_loss.set_title('Training performance')
    ax_loss.legend()
    # Plot feature importance
    ax_feature_important = plt.subplot(gs[1, :2])
    feature_importance = pd.DataFrame({
        'features': clf.feature_name(), 'importance': clf.feature_importance()
        }).sort_values('importance', ascending=False)
    ax_feature_important.bar(x=np.arange(len(feature_importance)),
                             height=feature_importance['importance'])
    ax_feature_important.set_xticks(np.arange(len(feature_importance)))
    ax_feature_important.set_xticklabels(feature_importance['features'])
    ax_feature_important.set_ylabel(
        'Feature importance (# times used to split)')
    ax_feature_important.set_title('Feature importance')

    # plot decision tree
    ax_decision_tree = plt.subplot(gs[2:, :-1])
    lgb.plot_tree(clf, ax=ax_decision_tree)
    plt.show()


def show_model_performance(gbm, evals_result):
    # show model importance
    # lgb.plot_importance(gbm)
    # Show Decision
    # lgb.plot_tree(gbm, figsize=(20,8), show_info=['split_gain'])
    # graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
    # graph.render(view=True)
    lgb.plot_metric(evals_result, config.metric)
    plt.show()


def record_performance(y_test, predicted):
    log.info('score is: {}\n'.format(accuracy_score(y_test, predicted)))
    log.info(' \n {}'.format(classification_report(y_test, predicted,
                                                   target_names=['Died',
                                                                 'Survived'])))
    log.info("------------------------------\n")


if __name__ == "__main__":
    main()
