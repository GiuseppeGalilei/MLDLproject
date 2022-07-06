"""
Based on the implementation in https://github.com/AshwinRJ/Federated-Learning-PyTorch
"""

import numpy as np
import collections

def _cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_cls_count = {i: [0 for _ in range(10)] for i in range(num_users)}
    targets = dataset.targets
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        for e in dict_users[i]:
            dict_users_cls_count[i][targets[e]] += 1
    return dict_users, dict_users_cls_count


def another_cifar_iid(server_id, server_labels, num_users):
    users = np.arange(0, num_users)
    num_items_balanced = int(len(server_id) / num_users)
    dict_users = collections.defaultdict(dict)
    labels = np.arange(0, 10)
    dict_labels = get_dict_labels(server_id, server_labels)
    all_idxs = [i for i in range(len(server_id))]
    new_dict = {}
    nets_cls_counts = collections.defaultdict(dict)
    for user in users:
        for label in labels:
            dict_users[user][label] = set(np.random.choice(dict_labels[label],
                                                           int(num_items_balanced / 10), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[user][label])
            nets_cls_counts[user][label] = len(list(dict_users[user][label]))
        new_dict[user] = set().union(dict_users[user][0], dict_users[user][1], dict_users[user][2],
                                     dict_users[user][3], dict_users[user][4], dict_users[user][5], dict_users[user][6],
                                     dict_users[user][7], dict_users[user][8], dict_users[user][9])

    return server_labels, new_dict, nets_cls_counts


def _cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    targets = dataset.targets
    dict_users_cls_count = {i: [0 for _ in range(10)] for i in range(num_users)}

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        for e in dict_users[i]:
            dict_users_cls_count[i][targets[e.astype(int)]] += 1
    return dict_users, dict_users_cls_count


def _cifar_noniid_unbalanced(dataset, num_users):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    targets = dataset.targets
    dict_users_cls_count = {i: [0 for _ in range(10)] for i in range(num_users)}

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 20

    random_shard_size = np.random.randint(min_shard, max_shard+1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)
    
    sizes = random_shard_size.tolist()
    cnt = 0
    for i in range(len(sizes)):
        if sizes[i] == 0:
            sizes[i] = 1
            cnt += 1
    while cnt > 0:
        i = np.random.randint(num_users)
        if sizes[i] > 1:
            sizes[i] -= 1
            cnt -= 1
            
    random_shard_size = np.array(sizes)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        random_shard_size = random_shard_size-1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                
    for i in range(num_users):
        for e in dict_users[i]:
            dict_users_cls_count[i][targets[e.astype(int)]] += 1

    return dict_users, dict_users_cls_count

def get_server(train_dataset):
    server_data = []
    server_id = []
    server_labels = []
    for i in range(len(train_dataset)):
        server_data.append(train_dataset[i][0])
        server_labels.append(train_dataset[i][1])
        server_id.append(i)
    return server_data, server_labels, server_id


def get_dict_labels(server_id, server_labels):
    dict_labels = {}
    num_classes = 10
    labels = np.arange(0, num_classes)  # the 10 classes we have : from 0 to 9
    for label in labels:
        if label not in dict_labels:
            dict_labels[label] = []
    for i in range(len(server_labels)):
        for label in labels:
            if label == server_labels[i]:
                dict_labels[label].append(server_id[i])
    return dict_labels

def get_another_user_groups(dataset, iid=True, unbalanced=False, tot_users=100):
    user_groups = None
    if iid:
        server_data, server_labels, server_id = get_server(dataset)
        server_labels, user_groups, dict_user_cls_count = another_cifar_iid(server_id, server_labels, tot_users)
    else:
        if unbalanced:
            user_groups, dict_user_cls_count = _cifar_noniid_unbalanced(dataset, tot_users)
        else:
            user_groups, dict_user_cls_count = _cifar_noniid(dataset, tot_users)

    return user_groups, dict_user_cls_count


def get_user_groups(dataset, iid=True, unbalanced=False, tot_users=100):
    user_groups = None
    if iid:
        user_groups, dict_user_cls_count = _cifar_iid(dataset, tot_users)
    else:
        if unbalanced:
            user_groups, dict_user_cls_count = _cifar_noniid_unbalanced(dataset, tot_users)
        else:
            user_groups, dict_user_cls_count = _cifar_noniid(dataset, tot_users)

    return user_groups, dict_user_cls_count
