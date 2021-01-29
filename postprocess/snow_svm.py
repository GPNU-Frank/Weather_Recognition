import torch
import os
import numpy as np




def main():
    # 支持向量机预测
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    svc.score(x_train, y_train)
    train_acc_svc = accuracy_score(svc.predict(x_train), y_train)
    test_acc_svc = accuracy_score(svc.predict(x_test), y_test)
    print(train_acc_svc, test_acc_svc)
    svc.predict(x_test)


if if __name__ == "__main__":
    main()