import pandas as pd
import numpy as np
import os
import lightgbm as lgb



def load_data(filename, sep=","):
    train = pd.read_csv(filename, sep=sep)
    return train


def select_feature(train_data):
    selected_columns = ["Age", "Sex"]
    train = train_data[selected_columns]
    return train


def binary_sex(train_data):
    train_data.Sex = train_data.Sex.map(lambda sex: 0 if 'male' == sex else 1)
    return train_data


def handle_na(train_data):
    return train_data.dropna()


def train_model(train_data, labels):
    print("train model...")
    classifier = lgb.LGBMClassifier()
    return classifier.fit(train_data, labels)


def get_result_and_save(classfier):
    print("predict...")
    test = load_data("test.csv", ",")
    test = binary_sex(test)
    res = classfier.predict(select_feature(test))
    print("save...")
    submission = pd.DataFrame({
                                  'PassengerId': test['PassengerId'],
                                  'Survived': res
                                  })
    submission.to_csv("submission.csv", index=False)


def main():
    # TODO do something
    train = load_data("train.csv")
    train = handle_na(train)
    labels = train['Survived']
    train = select_feature(train)
    train = binary_sex(train)
    model = train_model(train, labels)
    get_result_and_save(model)


if __name__ == "__main__":
    main()
