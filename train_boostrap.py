import random
import numpy as np
import pandas as pd
import time
import math
from boostrap import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def main():
    num_runs = int(input("Nhập số lần muốn test: "))
    total_time_code = 0
    total_acc_code = 0
    total_time_skl = 0
    total_acc_skl = 0

    filename = "D:\\data\\DLBCL 2.csv"
    data = pd.read_csv(filename)
    X = data.iloc[:, 1:].values.astype(float)
    y = data.iloc[:, 0].values.astype(np.int64)
    
    n_feature = int(math.sqrt(len(data.columns) - 1))
    for _ in tqdm(range(num_runs)):
        number_random = np.random.randint(1, 1000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=number_random
        )
        s_t_c = time.time()
        
        
        
        RF = RandomForest(n_estimators=100, max_features=n_feature, max_depth=10, min_samples = 2)
        
        
        RF.fit(X_train, y_train)
        
        print(RF.predict(X_test))
        predictions = RF.score(X_test,y_test)
        
        
        total_acc_code += predictions
        
        
        
        e_t_c = time.time()
        total_time_code += e_t_c - s_t_c

        s_t_skl = time.time()
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, max_features=n_feature, random_state=number_random
        )
        model.fit(X_train, y_train)
        total_acc_skl += model.score(X_test, y_test)
        e_t_skl = time.time()
        total_time_skl += e_t_skl - s_t_skl

    accuracy_code = total_acc_code / num_runs
    total_time_code = total_time_code / num_runs

    accuracy_skl = total_acc_skl / num_runs
    total_time_skl = total_time_skl / num_runs

    print(f"\nXác xuất code: {accuracy_code}")
    print(f"Xác xuất skl: {accuracy_skl}")
    print(f"Time code:{total_time_code}")
    print(f"Time skl: {total_time_skl}")


if __name__ == "__main__":
    main()
    
