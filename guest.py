# role: guest


from operator import index
from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable
from i_o.utils import read_from_csv_with_no_lable
from ml.feature.instance import Instance
from federation.transfer_inst import TransferInstGuest
from sklearn.metrics import roc_auc_score
from ml.utils import consts
import os
import numpy as np
import pandas as pd
import time

def alignment():
    train = pd.read_csv("./data/guest_train.csv")
    print(train.head())
    train_sorted = train.sort_values(by='id', ascending=True)
    print(train_sorted.head())
    train_sorted.to_csv("./data/guest_train_sorted.csv", index=False)


def hetero_secure_boost_guest_main():

    alignment()

    port = 11111
    # 实际传参过程
    
    # transfer
    transfer_inst = TransferInstGuest(port=port, conn_num=1)

    # guest, 设置相关参数
    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # ******************文件读取和交集处理******************
    # 读取交集list
    train_data_path = os.path.join("./data/guest_train_sorted.csv")
    intersection_list = np.loadtxt("./psi/data_A_result.csv", delimiter=",", dtype=int)
    intersect_id = set(intersection_list)
    # 取数据
    header, ids, features, lables = read_from_csv_with_lable(train_data_path)
    train_instances = []
    for i, feature in enumerate(features):
        if int(ids[i]) not in intersect_id:
            continue
        inst = Instance(inst_id=ids[i], features=feature, label=lables[i])
        train_instances.append(inst)
    print("train_instances number: ",len(train_instances))
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header

    # test_data
    # test_data_path = os.path.join("./data/guest_test.csv")
    # header, ids, features = read_from_csv_with_no_lable(test_data_path)
    # test_instances = []
    # for i, feature in enumerate(features):
    #     inst = Instance(inst_id=ids[i], features=feature)
    #     test_instances.append(inst)
    # print("test_instances number: ",len(test_instances))
    # test_instances = DTable(False, test_instances)
    # test_instances.schema['header'] = header


    # fit 训练模型
    start = time.time()
    hetero_secure_boost_guest.fit(data_instances=train_instances)
    cost = time.time() - start

    # predict
    # predict_result = hetero_secure_boost_guest.predict(test_instances)
    # test_label = pd.read_csv("./data/test_label.csv")
    # predict_proba = []
    # def func(kvs):
    #     correct_num = 0
    #     for _, v in kvs:
    #         predict_proba.append(v[2])
    # accuracy = predict_result.mapPartitions(func)
    # print('auc is: ', roc_auc_score(test_label,predict_proba))

    # print("train cost:", cost)

    # output model
    model = hetero_secure_boost_guest.sync_model_from_host()
    np.save("model.npy", model, allow_pickle=True)
    

    

if __name__ == '__main__':
    hetero_secure_boost_guest_main()
