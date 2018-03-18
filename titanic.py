import sys

import lightgbm as lgb
import pandas as pd
import logging as log
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    )


# TODO 可以将名字的长度加入到预测中
# TODO 对某些难以预测的类别的某个值进行处理
def main():
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %('
                           'levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='tmp.log',
                    filemode='w')
    off_line_test()
    sys.exit(0)


def off_line_test():
    # 对数据进行处理, 分用于训练模型
    raw_train_data = load_data("train.csv")
    raw_train_data = handle_na(raw_train_data)
    get_offline_result(
        raw_train_data)  # train = select_feature(raw_train_data)  #  #  #  #
    #  train = map_feature(train)  # model = train_model(train)  #  #  #  #
    # get_result_and_save(model)


def get_offline_result(raw_train_data):
    train = select_feature(raw_train_data)
    train = map_feature(train)
    x_train, x_test, y_train, y_test = train_test_split(train,
                                                        raw_train_data.Survived,
                                                        test_size=0.2,
                                                        random_state=42)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    final_train_set = lgb.Dataset(label=raw_train_data.pop('Survived'),
                                  data=raw_train_data)
    lgb_params = {
        'boosting': 'dart', 'learing_rate': 0.01, 'min_data_in_leaf': 20,
        'feature_fraction': 0.7, 'metric': 'binary_logloss', 'drop_rate': 0.15,
        'application': 'binary',
        }
    evaluations = {}
    clf = lgb.train(params=lgb_params, train_set=train_data,
                    valid_sets=[train_data, test_data],
                    valid_names=['Train', 'Test'], evals_result=evaluations,
                    num_boost_round=500, early_stopping_rounds=100,
                    verbose_eval=20)
    optimal = clf.best_iteration
    fig, axs = plt.subplots(2, 1, figsize=[8, 8])

    # Plot the log loss during training
    axs[0].plot(evaluations['Train']['binary_logloss'], label='Train')
    axs[0].plot(evaluations['Test']['binary_logloss'], label='Test')
    axs[0].set_ylabel('Log loss')
    axs[0].set_xlabel('Boosting round')
    axs[0].set_title('Training performance')
    axs[0].legend()

    # Plot feature importance
    importances = pd.DataFrame({
        'features': clf.feature_name(), 'importance': clf.feature_importance()
        }).sort_values('importance', ascending=False)
    axs[1].bar(x=np.arange(len(importances)), height=importances['importance'])
    axs[1].set_xticks(np.arange(len(importances)))
    axs[1].set_xticklabels(importances['features'])
    axs[1].set_ylabel('Feature importance (# times used to split)')
    axs[1].set_title('Feature importance')

    plt.show()
    predicted = np.round(clf.predict(x_test))
    log.info('Accuracy score = \t {}'.format(accuracy_score(y_test, predicted)))
    log.info(
        'Precision score = \t {}'.format(precision_score(y_test, predicted)))
    log.info('Recall score =   \t {}'.format(recall_score(y_test, predicted)))
    log.info('F1 score =      \t {}'.format(f1_score(y_test, predicted)))

    test = pd.read_csv("test.csv", sep=",")
    Id = test.PassengerId
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare"]
    test = test[selected_columns]
    test = map_feature(test)
    res = list(map(lambda x: 1 if x == 1.0 else 0, np.round(clf.predict(test))))
    pd.DataFrame({
        'PassengerId': Id, 'Survived': res
        }).to_csv('submission.csv', index=False)


def load_data(filename, sep=","):
    train = pd.read_csv(filename, sep=sep)
    return train


def select_feature(train_data):
    # selected_columns = ["Age", "Sex", "Pclass", "Fare"]
    # selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare", "Survived"]
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare"]
    train = train_data[selected_columns]
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


def train_model(train_data):
    print("train model...")
    lgb_params = {
        'learning_rate': 0.01, 'max_depth': 6, 'num_leaves': 64, 'max_bin': 100,
        'objective': 'mse', 'label': ["Survived"],
        }
    # 'feature_name': train_data.columns,
    # classifier = lgb.LGBMRegressor(learning_rate=0.3, max_depth=3)
    # return classifier.fit(train_data, labels)
    data_train_lgb = lgb.Dataset(train_data)
    cv_model = lgb.cv(lgb_params, data_train_lgb)  # return array of loss
    classifier = lgb.train(lgb_params, data_train_lgb)
    return classifier


def get_result(classfier, test):
    print("predict result for specified data...")
    res = classfier.predict(test)
    res = list(map(lambda x: 1 if x > 0.5 else 0, res))  # TOD 比较预测结果和指定结果
    # res = pd.DataFrame
    return (test.Survived == res).sum() / len(test)


def get_result_and_save(classfier, filename="test.csv"):
    print("predict result for specified data...")
    test = load_data(filename, ",")
    Id = test['PassengerId']
    columns = ["Sex", "Age", "Embarked", "Pclass", "Fare"]
    test = test[columns]
    test = map_feature(test)
    res = classfier.predict(test)
    res = list(map(lambda x: 1 if x > 0.5 else 0, res))
    print("save...")
    submission = pd.DataFrame({
        'PassengerId': Id, 'Survived': res
        })
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
