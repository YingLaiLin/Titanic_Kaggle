import sys

import lightgbm as lgb
import pandas as pd


# TODO 写一个交叉验证框架
# TODO 可以将名字的长度加入到预测中
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


def train_model(train_data, labels):
    print("train model...")
    classifier = lgb.LGBMRegressor(learning_rate=0.3, max_depth=3)
    return classifier.fit(train_data, labels)


def get_result_and_save(classfier):
    print("predict...")
    test = load_data("test.csv", ",")
    Id = test['PassengerId']
    test = select_feature(test)
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
