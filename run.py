import os
import dfp_training
import pickle
import numpy as np
import sys
def dump_to_txt_files(pt_acc_list, acc_list):
    with open("ptacc_cifar_quantize_han.txt", "w") as f:
        for item in pt_acc_list:
            f.write("%s\n"%item)
    with open("acc_cifar_quantize_han.txt", "w") as f:
        for item in acc_list:
            f.write("%s\n"%item)

acc_list = []
pt_acc_list = []
count = 0
retrain = 0
parent_dir = './'
base_model = 'base_prune.pkl'




keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
central_value = {}
c_pos = {}
c_neg = {}
with open('./masks/' + base_model,'rb') as f:
    (weights_mask, biases_mask) = pickle.load(f)
with open('./weights/' + base_model, 'rb') as f:
    (weights_val, biases_val) = pickle.load(f)

for key in keys:
    central_value[key] = np.mean(weights_val[key]* weights_mask[key])
    pos_plus = np.logical_and((weights_val[key]* weights_mask[key]) > central_value[key], weights_mask[key])
    pos_minus = np.logical_and((weights_val[key]* weights_mask[key]) <= central_value[key], weights_mask[key])
    c_pos[key] = np.mean(weights_val[key][pos_plus])
    c_neg[key] = np.mean(weights_val[key][pos_minus])
    tmp = weights_val[key]* weights_mask[key]
    print('nozeros{}, total{}'.format((tmp!=0).sum(), len(tmp.flatten())))
# sys.exit()

print(central_value)
print(c_pos)
print(c_neg)
# quantisation_bits = [2, 4, 8, 16]
# 1 bit sign, 2 bits range
quantisation_bits = [4, 6, 8, 16, 32]
quantisation_bits = [4, 6, 8]
quantisation_bits = [item - 1 for item in quantisation_bits]
pcov = [0,0]
dynamic_range = 4
READ_ONLY = False
for q_width in quantisation_bits:
    # measure acc
    param = [
        ('-t', 0),
        ('-q_bits',q_width),
        ('-pretrain',1),
        ('-parent_dir', parent_dir),
        ('-base_model', base_model),
        ('-c_pos', c_pos),
        ('-c_neg', c_neg),
        ('-central_value', central_value),
        ('-read_only',READ_ONLY)
        ]
    pre_train_acc = dfp_training.main(param)
    param = [
        ('-t', 1),
        ('-q_bits',q_width),
        ('-pretrain',1),
        ('-parent_dir', parent_dir),
        ('-base_model', base_model),
        ('-c_pos', c_pos),
        ('-c_neg', c_neg),
        ('-central_value', central_value),
        ('-read_only',READ_ONLY)
        ]
    train_acc = dfp_training.main(param)
    train_acc = 0

    pt_acc_list.append(pre_train_acc)
    acc_list.append(train_acc)
    print(pt_acc_list)
    print(acc_list)
    dump_to_txt_files(pt_acc_list, acc_list)
    count = count + 1
print('accuracy summary: {}'.format(pt_acc_list))
print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
