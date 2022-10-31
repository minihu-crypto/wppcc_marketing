import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score


    

def get_feature(fid, fid_map, data):
    for fname in fid_map:
        if fid_map[fname] == fid:
            return data[fname]  

def predict(model_path = "model.npy", test_path = "./data/test.csv"):
    model = np.load(model_path, allow_pickle=True).item()
    test_data = pd.read_csv(test_path)
    # predict每个样本
    y_pred = []
    for i in range(test_data.shape[0]):
        data = test_data.iloc[i]
        # traverse，找到叶子结点nid即可
        result = 0
        for tree_id in range(len(model['trees'])):
            if 0 not in model['split_maskdicts_guest'][tree_id].keys():
                model['split_maskdicts_guest'][tree_id][0] = 0
            if 0 not in model['split_maskdicts_host'][tree_id].keys():
                model['split_maskdicts_host'][tree_id][0] = 0
            tree = model['trees'][tree_id]
            node_id = 0
            is_leaf = False
            while not is_leaf:
                node_bid = tree[node_id]['bid']
                fid = tree[node_id]['fid']
                if tree[node_id]['sitename']=='guest':
                    split_value = model['split_maskdicts_guest'][tree_id][node_bid]
                    if get_feature(fid, model['fid_map_guest'], data) <= split_value:
                        node_id = tree[node_id]['left_nodeid']
                    else:
                        node_id = tree[node_id]['right_nodeid']
                else:
                    split_value = model['split_maskdicts_host'][tree_id][node_bid]
                    if get_feature(fid, model['fid_map_host'], data) <= split_value:
                        node_id = tree[node_id]['left_nodeid']
                    else:
                        node_id = tree[node_id]['right_nodeid']
                if tree[node_id]['left_nodeid'] == -1:
                    is_leaf = True
            result += tree[node_id]['weight']
        y_pred.append(result)
    test_label = pd.read_csv("./data/test_label.csv")
    st = set()
    for y in y_pred:
        if y not in st:
            st.add(y)
    print(st)
    print(roc_auc_score(test_label, y_pred))


    
    


if __name__ == '__main__':
    # 传入model_path 和 test_path
    if len(sys.argv) > 1:
        predict(sys.argv[1], sys.argv[2])
    else:
        predict()