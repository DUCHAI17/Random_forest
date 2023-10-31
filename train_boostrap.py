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
    

#tieu_duong
#Xác xuất code: 0.6467532467532469
#Xác xuất skl: 0.7545454545454546
#Time code:2.104680395126343
#Time skl: 0.16433145999908447
#Time: kém gấp 10,5 lần



#data
#Xác xuất code: 0.5208333333333333
#Xác xuất skl: 0.7416666666666666
#Time code:0.8588677406311035
#Time skl: 0.12912168502807617
#Time: kém gấp 6,6 lần


#DLBCL
#Xác xuất code: 0.75625
#Xác xuất skl: 0.86875
#Time code:5.934693622589111
#Time skl: 0.1941629409790039
#Time: kém gấp 29,5 lần


#adenocarcinoma
#Xác xuất code: 0.825
#Xác xuất skl: 0.81875
#Time code:6.5916378736495975
#Time skl: 0.19154398441314696
#Time: kém gấp 33 lần


#colon
#Xác xuất code: 0.5769230769230769
#Xác xuất skl: 0.7538461538461537
#Time code:3.279830718040466
#Time skl: 0.1589811325073242
#Time: kém gấp 20,5 lần




#Prostate
#Xác xuất code: 0.4285714285714285
#Xác xuất skl: 0.880952380952381
#Time code:8.967152166366578
#Time skl: 0.25345277786254883
#Time: kém gấp 35,88 lần


#Leukemia_4c1
#Xác xuất code: 0.54
#Xác xuất skl: 0.78
#Time code:8.57255232334137
#Time skl: 0.2238736629486084
#Time: kém gấp 43 lần