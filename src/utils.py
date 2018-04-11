# -*- coding: utf-8 -*-
import pandas as pd

from src import config


def load_data(filename, sep=config.csv_seperator):
    return pd.read_csv(filename, sep=sep)


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


def handle_na(train_data):
    """
           对数据进行缺失值处理
    :param train_data:
    :return:
    """
    # 年龄 和 费用用平均值填充
    train_data.Age = train_data.Age.fillna(train_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(train_data.Fare.median())
    # 上船港口默认为 S
    train_data.Embarked = train_data.Embarked.fillna('S')
    return train_data


def extract_features(train_data):
    """
            特征工程
    :param train_data:
    :return:
    """
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
    """
            特征工程
    :param train_data:
    :return:
    """
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
