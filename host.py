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
import random
import sys
import os


def hetero_seucre_boost_host():

    train_data_path = os.path.join("data/sell/host_train.csv")
    test_data_path = os.path.join("data/sell/host_test.csv")
    guest_ip = '192.168.1.2'
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


    # 从文件读取数据
    # train_data
    header, ids, features = read_from_csv_with_no_lable(train_data_path)
    train_instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature)
        train_instances.append(inst)
    print("train_instances number: ",len(train_instances))
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header

    # test_data
    header, ids, features = read_from_csv_with_no_lable(test_data_path)
    test_instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature)
        test_instances.append(inst)
    print("test_instances number: ",len(test_instances))
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header
    
    
    # fit
    hetero_secure_boost_host.fit(data_instances=train_instances)


    # predict
    hetero_secure_boost_host.predict(test_instances)

    # save model
    hetero_secure_boost_host.sync_model_to_guest()




if __name__ == '__main__':
    hetero_seucre_boost_host()