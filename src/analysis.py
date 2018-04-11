
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


# columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
# 'Cabin', Embarked']
# TODO  添加对 Name 的分析.
def main():
    train_data, labels = get_data()
    # analysis_Name(train_data)
    # analysis_Pclass(train_data, [0, 1])
    # analysis_Fare(train_data, 30)
    # analysis_Age(train_data, 30)
    get_pearson(train_data)
    sys.exit(0)


def get_data(train_file_name=config.train_file_name, sep=config.csv_seperator):
    train_data = pd.read_csv(train_file_name, sep)
    labels = train_data.Survived
    del train_data['PassengerId']
    return train_data, labels


def get_pearson(train_data):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('Spectral', 30)
    cax = ax.imshow(train_data.corr(), interpolation='nearest', cmap=cmap)
    print(train_data.corr())
    ax.grid(True)
    plt.title('Titanic Feature Correlation')
    labels = train_data.columns
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax, ticks=[i * 0.15 for i in range(-6, 10)])
    plt.show()


def analysis_Name(train_data):
    print(train_data.describe(include=['O']))
    for data in train_data:
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_data['Title'], train_data['Sex'])


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


