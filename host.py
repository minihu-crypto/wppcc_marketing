# role：host
# start parameters：4
#   - the address of csv file
#   - the proportion of data divided
#   - the port of guest
#   - the run_time_idx, it should be different among hosts

from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_host import HeteroSecureBoostingTreeHost
from i_o.utils import read_from_csv_with_no_lable
from ml.feature.instance import Instance
from federation.transfer_inst import TransferInstHost
import os
import numpy as np
import pandas as pd

def alignment():
    train = pd.read_csv("./data/host_train.csv")
    print(train.head())
    train_sorted = train.sort_values(by='id', ascending=True)
    print(train_sorted.head())
    train_sorted.to_csv("./data/host_train_sorted.csv", index=False)


def hetero_seucre_boost_host():
    
    alignment()

    guest_ip = '127.0.0.1'
    port = 11111
    run_time_idx = 0

    # 实际传参过程
    

    # transfer
    transfer_inst = TransferInstHost(ip=guest_ip, port=port)

    # host, 设置相关参数
    hetero_secure_boost_host = HeteroSecureBoostingTreeHost()
    hetero_secure_boost_host._init_model(hetero_secure_boost_host.model_param)
    hetero_secure_boost_host.set_transfer_inst(transfer_inst)
    hetero_secure_boost_host.set_runtime_idx(run_time_idx)


    # ******************文件读取和交集处理******************
    # 读取交集list
    train_data_path = os.path.join("./data/host_train_sorted.csv")
    intersection_list = np.loadtxt("./psi/data_B_result.csv", delimiter=",", dtype=int)
    intersect_id = set(intersection_list)
    # train_data
    header, ids, features = read_from_csv_with_no_lable(train_data_path)
    train_instances = []
    for i, feature in enumerate(features):
        if int(ids[i]) not in intersect_id:
            continue
        inst = Instance(inst_id=ids[i], features=feature)
        train_instances.append(inst)
    print("train_instances number: ",len(train_instances))
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header

    # test_data
    # test_data_path = os.path.join("./data/host_test.csv")
    # header, ids, features = read_from_csv_with_no_lable(test_data_path)
    # test_instances = []
    # for i, feature in enumerate(features):
    #     inst = Instance(inst_id=ids[i], features=feature)
    #     test_instances.append(inst)
    # print("test_instances number: ",len(test_instances))
    # test_instances = DTable(False, test_instances)
    # test_instances.schema['header'] = header
    
    
    # fit
    hetero_secure_boost_host.fit(data_instances=train_instances)


    # predict
    # hetero_secure_boost_host.predict(test_instances)

    # save model
    hetero_secure_boost_host.sync_model_to_guest()




if __name__ == '__main__':
    hetero_seucre_boost_host()