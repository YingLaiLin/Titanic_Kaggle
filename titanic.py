import sys

import lightgbm as lgb
import pandas as pd
import logging as log
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, precision_score,
    f1_score,
    )


# TODO 引入交叉验证选择最佳模型
def main():
    log.basicConfig(level=log.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %('
                           'levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='tmp.log',
                    filemode='a+')
    off_line_test()
    sys.exit(0)


def off_line_test():
    # 对数据进行处理, 分用于训练模型
    raw_train_data = load_data("train.csv")
    test_data = load_data("test.csv")
    train_data = pd.concat([raw_train_data, test_data])
    train_data = handle_na(train_data)
    get_offline_result(train_data, len(raw_train_data))


def get_offline_result(raw_train_data, labeld_size):
    train = select_features(raw_train_data)
    train = map_feature(train)
    x_train, x_test, y_train, y_test = train_test_split(train[:labeld_size],
                                                        raw_train_data[
                                                        :labeld_size].Survived,
                                                        test_size=0.20,
                                                        random_state=42)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    labels = raw_train_data[:labeld_size].Survived
    final_train_set = select_features(raw_train_data[:labeld_size])
    final_train_set = map_feature(final_train_set)
    final_train_set = lgb.Dataset(data=final_train_set, label=labels,
                                  free_raw_data=False)
    loss_evaluation = ['auc', 'binary_logloss']
    lgb_params = {
        'boosting_type': 'dart', 'learning_rate': 0.05, 'max_depth': -1,
        'application': 'binary',  # 'nthread': 5,
        'num_leaves': 64, 'max_bin': 512, 'metric': loss_evaluation,
        'subsample_for_bin': 200, 'subsample': 1, 'subsample_freq': 1,
        'colsample_bytree': 0.8, 'reg_alpha': 5, 'reg_lambda': 10,
        'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 5,
        'scale_pos_weight': 1, 'num_class': 1, 'drop_rate': 0.15,
        'feature_fraction': 0.7,
        }
    gridParams = {
        'learning_rate': [0.05], 'n_estimators': [8, 10, 16, 24],
        'num_leaves': [6, 8, 12, 16], 'boosting_type': ['dart'],
        'objective': ['binary'], 'random_state': [501],  # Updated from 'seed'
        'colsample_bytree': [0.64, 0.65, 0.66], 'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2], 'reg_lambda': [1, 1.2, 1.4],
        }
    evaluations = {}

    # clf = lgb.LGBMClassifier(params=lgb_params, train_set=train_data,
    #                 valid_sets=[train_data, test_data],
    #                 valid_names=['Train', 'Test'], evals_result=evaluations,
    #                 num_boost_round=1000, early_stopping_rounds=100,
    #                 verbose_eval=50)

    clf = lgb.LGBMRegressor(boosting_type='dart', objective='binary',
                             nthread=5, silent=True, max_depth=-1, max_bin=128,
                             subsample_for_bin=500, subsample=1,
                             subsample_freq=1, min_split_gain=0.5,
                             min_child_weight=1, min_child_samples=5,
                             scale_pos_weight=1)
    grid = GridSearchCV(clf, gridParams, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    predicted = np.round(grid.predict(x_test))
    log.info('grid best score is: {}\n'.format(grid.best_score_))
    log.info('score is: {}\n' .format(accuracy_score(y_test, predicted)))
    log.info(' \n {}'.format(
        classification_report(y_test, predicted,
                              target_names=['Died','Survived'])))
    log.info("------------------------------\n")  #


    #### retrain
    params = {
        'boosting_type': 'dart', 'max_depth': -1, 'objective': 'binary',
        'nthread': 5, 'silent': True,
        'num_leaves': grid.best_params_['num_leaves'],
        'learning_rate': grid.best_params_['learning_rate'], 'max_bin': 512,
        'subsample_for_bin': 200, 'subsample': grid.best_params_['subsample'],
        'subsample_freq': 1,
        'colsample_bytree': grid.best_params_['colsample_bytree'],
        'reg_alpha': grid.best_params_['reg_alpha'],
        'reg_lambda': grid.best_params_['reg_lambda'], 'min_split_gain': 0.5,
        'min_child_weight': 1, 'min_child_samples': 5, 'scale_pos_weight': 1,
        'num_class': 1, 'metric': 'binary_error'
        }
    gbm = lgb.train(params, train_set=train_data, num_boost_round=10000,
                    valid_sets=[train_data, test_data],
                    early_stopping_rounds=50, verbose_eval=50)

    # Plot importance
    lgb.plot_importance(gbm)

    plt.show()
    evaluate_and_save(gbm, train, labeld_size, raw_train_data[
                                               labeld_size:].PassengerId)  #
    # params['max_bin'] = grid.best_params_['max_bin']

    # visualize_loss(clf, evaluations, loss_evaluation)  # Search best
    # paramters via grid search  # grid = GridSearchCV(clf, gridParams,
    # verbose=1, cv=4, n_jobs=-1)  # grid = GridSearchCV(clf, gridParams,
    # verbose=1, cv=4, n_jobs=-1)  # grid.fit(x_train, y_train)  #
    # lgb_params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    #  lgb_params['learning_rate'] = grid.best_params_['learning_rate']  # #
    # params['max_bin'] = grid.best_params_['max_bin']  # lgb_params[
    # 'num_leaves'] = grid.best_params_['num_leaves']  # lgb_params[
    # 'reg_alpha'] = grid.best_params_['reg_alpha']  # lgb_params[
    # 'reg_lambda'] = grid.best_params_['reg_lambda']  # lgb_params[
    # 'subsample'] = grid.best_params_['subsample']

    # predicted = np.round(clf.predict(x_test))  # get the final model
    #  # optimal_boost_rounds = clf.best_iteration  # # clf_final =
    # lgb.train(train_set=final_train_set, params=lgb_params,  # #
    #             init_model=clf,  # # num_boost_round=optimal_boost_rounds,
    #  #                       verbose_eval=50)  # clf_final = lgb.train(
    # train_set=final_train_set, params=lgb_params,  #
    # init_model=clf, verbose_eval=50)  # evaluate_and_save(clf_final, train,
    #  labeld_size,  #                   raw_train_data[
    # labeld_size:].PassengerId)


def load_data(filename, sep=","):
    return pd.read_csv(filename, sep=sep)


def select_features(train_data):
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
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare", "Alone",
                        "Title"]
    train = train_data[selected_columns]
    # 缺失姓名的值默认为 0
    train_data.Title = train_data.Title.fillna(0)
    return train


def map_feature(train_data):
    columns = train_data.columns
    # 直接利用哑变量方法来对变量进行处理, 也可以通过 map 来进行变换

    # 因为 Fare 是一些连续值, 难以用于预测, 故将按照4分位数分为4个区间.[0,.25,.5,.75,1.]
    if "Fare" in columns:
        train_data.Fare = pd.qcut(train_data.Fare, q=4, labels=False)
    if "Age" in columns:
        train_data.Age = pd.qcut(train_data.Age, q=4, labels=False)
    data_dum = pd.get_dummies(train_data, drop_first=True)
    return data_dum


def handle_na(train_data):
    # 年龄 和 费用用平均值填充
    train_data.Age = train_data.Age.fillna(train_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(train_data.Fare.median())
    # 上船港口默认为 S
    train_data.Embarked = train_data.Embarked.fillna('S')

    return train_data


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
    importances = pd.DataFrame({
        'features': clf.feature_name(), 'importance': clf.feature_importance()
        }).sort_values('importance', ascending=False)
    ax_feature_important.bar(x=np.arange(len(importances)),
                             height=importances['importance'])
    ax_feature_important.set_xticks(np.arange(len(importances)))
    ax_feature_important.set_xticklabels(importances['features'])
    ax_feature_important.set_ylabel(
        'Feature importance (# times used to split)')
    ax_feature_important.set_title('Feature importance')

    # plot decision tree
    ax_decision_tree = plt.subplot(gs[2:, :-1])
    lgb.plot_tree(clf, ax=ax_decision_tree)
    plt.show()


def evaluate_and_save(clf, train, labeled_size, Ids):
    test = train[labeled_size:]
    res = list(map(lambda x: 1 if x == 1.0 else 0, np.round(
        clf.predict(test, num_iteration=clf.best_iteration))))

    pd.DataFrame({
        'PassengerId': Ids, 'Survived': res
        }).to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
