import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
# 'Cabin', Embarked']
def main():
    train_data, labels = get_data()
    analysis_Pclass(train_data, [0,1])


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
        plt.bar(index + bar_width * value_index,
                train_data[train_data.Survived == values[value_index]].Pclass.value_counts(),
                bar_width)

    plt.xlabel('Pclass level')
    plt.ylabel('number')
    plt.legend(["Survived", "not Survived"])
    plt.show()


if __name__ == "__main__":
    main()
