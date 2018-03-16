import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
# 'Cabin', Embarked']
def main():
    train_data, labels = get_data()
    # analysis_Pclass(train_data, [0, 1])
    analysis_Fare(train_data, 30)
    # analysis_Age(train_data, 30)
    sys.exit(0)

def get_data():
    train_data = pd.read_csv("train.csv")
    labels = train_data.Survived
    del train_data['PassengerId']
    return train_data, labels


"""
    分析 Pclass 是否对 Survived 造成影响
"""


def analysis_Pclass(train_data, groups, values):
    index = np.arange(groups)
    bar_width = 0.35
    for value_index in range(len(values)):
        plt.bar(index + bar_width * value_index, train_data[
            train_data.Survived == values[value_index]].Pclass.value_counts(),
                bar_width)

    plt.xlabel('Pclass level')
    plt.ylabel('count')
    plt.legend(["Survived", "not Survived"])
    plt.show()


def analysis_Age(train_data, num_bins):
    survived = train_data[train_data.Survived == 1].Age.dropna()
    not_survived = train_data[train_data.Survived == 0].Age.dropna()
    data = [survived, not_survived]
    plt.hist(data, num_bins)
    plt.xlabel('Age')
    plt.ylabel('count')
    plt.legend(["Survived", "not Survived"])
    plt.show()


def analysis_Fare(train_data, num_bins):
    # survived = train_data[train_data.Survived == 1].Fare
    # not_survived = train_data[train_data.Survived == 0].Fare
    # data = [survived, not_survived]
    train_data.groupby("Survived").Fare.hist(alpha=0.6)
    # plt.hist(data, num_bins, density=1)
    # plt.xlabel('Fare')
    # plt.ylabel('count')
    plt.legend(["not Survived", "Survived"])
    plt.show()


if __name__ == "__main__":
    main()
