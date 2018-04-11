# -*- coding: utf-8 -*-

import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from src import config
from src import utils

# TODO 引入 stacking
# TODO 引入自动存储和更换最佳超参数的功能
# TODO 引入训练时,改变学习率或者其他参数的机制
# 在 train 函数中 添加新的参数callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5
#  + [0.6] * 5)])
# decay learning_rates=lambda iter: 0.05 * (0.99 ** iter),
""" 运行时改变measure
    def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print('Add a new valid dataset at iteration 5...')
            env.model.add_valid(lgb_eval_new, 'new valid')
    callback.before_iteration = True
    callback.order = 0
    return callback
"""


def make_stacking(train_data, labeled_size, labels, Ids):
    train_set = train_data[:labeled_size]
    train_labels = labels[:labeled_size]
    test_set = train_data[labeled_size:]
    kf = KFold(n_splits=config.stacking_folds)
    params = {
        'n_estimators': config.n_estimators, 'random_state': 501,
        }
    models = [svm.SVC(), ExtraTreesClassifier(**params),
              RandomForestClassifier(**params), AdaBoostClassifier(**params),
              GradientBoostingClassifier(**params,
                                         learning_rate=config.learning_rate)]

    model_index = 0
    predictions = np.zeros((0, 1))
    test_predictions = np.zeros((len(train_set)))
    for train_index, test_index in kf.split(train_set):
        X_train, X_test = train_set.iloc[train_index], train_set.iloc[
            test_index]
        y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[
            test_index]
        clf = models[model_index].fit(X_train, y_train)
        print(clf.__class__, ' score: ',
              models[model_index].score(X_train, y_train))
        predictions = np.append(predictions, clf.predict(X_test))
        test_predictions = test_predictions + clf.predict(train_set)
        model_index += 1

    test_predictions = np.round(test_predictions / config.stacking_folds)
    stakcing_model = RandomForestClassifier(**params)
    stakcing_clf = stakcing_model.fit(predictions.reshape(-1, 1),
                                      test_predictions.reshape(-1, 1))
    model = train_model(train_data, labeled_size, labels)
    prediction = predict(model, train_data, labeled_size)
    prediction = np.array(prediction).reshape(-1, 1)
    stacking_prediction = stakcing_clf.predict(prediction)
    save_prediction(None, np.array(stacking_prediction).astype(int),
                    labeled_size, Ids)


def main():
    # config_log()
    train_data, labeled_size, labels, Ids = utils.pre_process_data()
    if config.can_grid_search_hyper_params:
        search_object = search_hyper_parameters(train_data, labeled_size,
                                                labels)
        if config.can_save_best_params:
            save_best_params(search_object)
    if config.can_stacking:
        make_stacking(train_data, labeled_size, labels, Ids)
    else:
        model = train_model(train_data, labeled_size, labels)
        prediction = predict(model, train_data, labeled_size)
        save_prediction(model, prediction, labeled_size, Ids)
    sys.exit(0)


def search_hyper_parameters(train_data, labeled_size, labels):
    """
           通过 GridSearch 查找超参数
    :param train_data:
    :param labeled_size:
    :param labels:
    :return:
    """
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
        'bagging_fraction': config.cv_bagging_fraction,
        'bagging_freq': config.cv_bagging_freq,
        }
    # clf = lgb.LGBMRegressor(boosting_type='dart', objective='binary')
    clf = lgb.LGBMClassifier(boosting_type='dart', objective='binary')
    grid = GridSearchCV(clf, gridParams, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid


def save_best_params(search_object, filename=config.params_file_name,
                     mode=config.params_file_mode):
    with open(filename, mode) as writer:
        writer.write('best params after grid search\n')
        for param_key in search_object.best_params_:
            writer.write("{} : {}\n".format(param_key,
                                            search_object.best_params_[
                                                param_key]))
        writer.write("------------------------------\n")


def train_model(train, labeled_size, labels):
    params = {
        'boosting_type': config.boosting_type, 'objective': 'binary',
        'metric': config.metric, 'learning_rate': config.learning_rate,
        # 根据 GridSearchCV 中的参数进行设置
        'nthread': config.n_threads, 'n_estimators': config.n_estimators,
        'feature_fraction': 0.9, 'random_state': 501,
        'lambda_l1': config.lambda_l1, 'lambda_l2': config.lambda_l2,
        'max_depth': -1, 'num_leaves': config.num_leaves,
        'bagging_fraction': config.bagging_fraction,
        'bagging_freq': config.bagging_freq,
        }
    x_train, x_test, y_train, y_test = train_test_split(train[:labeled_size],
                                                        labels[:labeled_size],
                                                        test_size=config.cv_split_size,
                                                        random_state=42)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    eval_results = {}

    gbm = lgb.train(params, train_set=train_data, num_boost_round=2000,
                    valid_sets=[train_data, test_data],
                    evals_result=eval_results, verbose_eval=50,
                    early_stopping_rounds=config.early_stopping_rounds)
    predicted = np.round(gbm.predict(x_test))
    record_performance(y_test, predicted)
    show_model_performance(gbm, eval_results)
    return gbm


def predict(model, train_data, labeled_size):
    test = train_data[labeled_size:]
    return list(map(lambda x: 1 if x == 1.0 else 0, np.round(
        model.predict(test, num_iteration=model.best_iteration))))


def save_prediction(clf, prediction, labeled_size, Ids):
    # 存储模型参数
    if clf is not None:
        clf.save_model(config.tree_params_file_name)
    # 存储预测结果
    pd.DataFrame({
        'PassengerId': Ids[labeled_size:], 'Survived': prediction
        }).to_csv(config.submission_file_name, index=False)


def show_model_performance(gbm, evals_result):
    # show model importance
    # lgb.plot_importance(gbm)
    # Show Decision Tree
    if config.can_plot_tree:
        graph = lgb.create_tree_digraph(gbm, name='Decision Tree')
        graph.render(view=True)
    if config.can_show_metric:
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        for index in range(len(config.metric)):
            lgb.plot_metric(evals_result, config.metric[index],
                            title=config.metric[index], ax=axs[index])
    plt.show()


def record_performance(y_test, predicted, filename=config.performance_file_name,
                       mode=config.performance_file_mode):
    with open(filename, mode) as writer:
        writer.write('score is: {}\n'.format(accuracy_score(y_test, predicted)))
        writer.write(' \n {}'.format(classification_report(y_test, predicted,
                                                           target_names=['Died',
                                                                         'Survived'])))
        writer.write("------------------------------\n")


if __name__ == "__main__":
    main()
