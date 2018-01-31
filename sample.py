import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import svm


def main():
    iris_data = datasets.load_iris()
    train = iris_data["data"][:100]
    target = iris_data["target"][:100]

    f1_scores = np.array([])
    for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True).split(train, target):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        model = svm.SVM()
        model.fit(X_train, y_train)
        
        predict = model.predict(X_test)
        f1_scores = np.append(f1_scores, calc_F1(predict, y_test))

    print("F-measure : ", f1_scores)
    print("F-measure average : ", f1_scores.mean())    


def calc_F1(predict, y):
    precision = calc_precision(predict, y)
    recall = calc_recall(predict, y)
    
    return np.float((2 * recall * precision) / (recall + precision))


def calc_precision(predict, y):
    tp = np.float(sum([1 for i, j in zip(predict, y) if i == 1 and j == 1]))
    tp_fp = np.float(sum(predict))
    
    return tp / tp_fp


def calc_recall(predict, y):
    tp = np.float(sum([1 for i, j in zip(predict, y) if i == 1 and j == 1]))
    tp_fn = np.float(sum(y))
    
    return tp / tp_fn


if __name__ == '__main__':
    main()