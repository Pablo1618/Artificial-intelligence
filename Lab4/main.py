import numpy as np
import time

from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic


def main():
    np.random.seed(123)
    #np.random.seed(int(time.time()))

    train_data, test_data = load_titanic()

    dt = DecisionTree({"depth": 14}) # 14
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)

    rf = RandomForest({"ntrees": 8, "feature_subset": 4, "depth": 16}) # 10 2 14
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)

    #0.86
    #0.78
    #0.91   10 4 16
    #0.84
if __name__=="__main__":
    main()