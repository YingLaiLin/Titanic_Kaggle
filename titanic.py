import sys

import lightgbm as lgb
import pandas as pd
import logging as log


# TODO 将训练集和测试集整合
# TODO 可以将名字的长度加入到预测中
# TODO train 和 fit 的区别是什么.
# TODO 对某些难以预测的类别的某个值进行处理
def main():
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %('
                           'levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='tmp.log',
                    filemode='w')
    folds = 5
    off_line_test(folds)
    sys.exit(0)


def off_line_test(folds):
    # 对数据进行处理, 分用于训练模型
    raw_train_data = load_data("train.csv")
    raw_train_data = handle_na(raw_train_data)
    get_offline_result(raw_train_data, folds)
    train = select_feature(raw_train_data)
    train = map_feature(train)
    model = train_model(train)
    get_result_and_save(model)


def get_offline_result(raw_train_data, folds):
    split_pos = len(raw_train_data) // folds
    for cnt in range(folds):
        train = raw_train_data.sample(frac=1.0)
        train = select_feature(train)
        train = map_feature(train)
        cv_train = train.iloc[split_pos:]
        test = train.iloc[:split_pos]
        model = train_model(cv_train)
        # 对测试数据进行预测并存储
        precision = get_result(model, test)
        log.info("%d round, precision is %.6f" % (cnt, precision))


def load_data(filename, sep=","):
    train = pd.read_csv(filename, sep=sep)
    return train


def select_feature(train_data):
    # selected_columns = ["Age", "Sex", "Pclass", "Fare"]
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare", "Survived"]
    train = train_data[selected_columns]
    return train


def map_feature(train_data):
    columns = train_data.columns
    # 直接利用哑变量方法来对变量进行处理, 也可以通过 map 来进行变换
    data_dum = pd.get_dummies(train_data, drop_first=True)
    # 因为 Fare 是一些连续值, 难以用于预测, 故将按照4分位数分为4个区间.[0,.25,.5,.75,1.]
    if "Fare" in columns:
        data_dum.Fare = pd.qcut(data_dum.Fare, q=4, labels=False)
    if "Age" in columns:
        data_dum.Age = pd.qcut(data_dum.Age, q=4, labels=False)

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
        'learning_rate': 0.01, 'max_depth': 6, 'num_leaves': 64,
        'max_bin': 100, 'objective': 'mse',
        'label': ["Survived"],
        }
    # 'feature_name': train_data.columns,
    # classifier = lgb.LGBMRegressor(learning_rate=0.3, max_depth=3)
    # return classifier.fit(train_data, labels)
    data_train_lgb = lgb.Dataset(train_data)
    lgb.cv(lgb_params, data_train_lgb)  # return array of loss
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
