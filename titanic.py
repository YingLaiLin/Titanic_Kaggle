import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import sys


# TODO 增加对缺失值的处理
# TODO 写一个交叉验证框架
# TODO 尝试对 lightGBM 学习参数的调整


def main():
    # 对数据进行处理, 分用于训练模型
    train = load_data("train.csv")
    train = handle_na(train)
    labels = train['Survived']
    train = select_feature(train)
    train = map_feature(train)
    model = train_model(train, labels)
    # 对测试数据进行预测并存储
    get_result_and_save(model)
    sys.exit(0)


def load_data(filename, sep=","):
    train = pd.read_csv(filename, sep=sep)
    return train


def select_feature(train_data):
    # selected_columns = ["Age", "Sex", "Pclass", "Fare"]
    selected_columns = ["Sex", "Age", "Embarked", "Pclass", "Fare"]
    train = train_data[selected_columns]
    return train


def map_feature(train_data):
    columns = train_data.columns
    # 将性别映射为 [0,1]
    if "Sex" in columns:
        train_data.Sex = train_data.Sex.map(
            lambda sex: 0 if 'male' == sex else 1)
    # 因为 Fare 是一些连续值, 难以用于预测, 故将其分为 4个 区间.
    if "Fare" in columns:
        train_data.Fare = pd.qcut(train_data.Fare, q=4, labels=False)
    if "Age" in columns:
        train_data.Age = pd.qcut(train_data.Age, q=4, labels=False)
    if "Embarked" in columns:
        train_data.Embarked = train_data.Embarked.map(
            lambda embarked: ord(embarked) - 65)
    return train_data


def handle_na(train_data):
    # 年龄 和 费用用平均值填充
    train_data.Age = train_data.Age.fillna(train_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(train_data.Fare.median())
    # 上船港口默认为 S
    train_data.Embarked = train_data.Embarked.fillna('S')
    return train_data


def train_model(train_data, labels):
    print("train model...")
    classifier = lgb.LGBMClassifier(learning_rate=0.3, max_depth=3)
    return classifier.fit(train_data, labels)


def get_result_and_save(classfier):
    print("predict...")
    test = load_data("test.csv", ",")
    test = map_feature(test)
    res = classfier.predict(select_feature(test))
    res = list(map(lambda x: 1 if x > 0.5 else 0, res))
    print("save...")
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'], 'Survived': res
        })
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
