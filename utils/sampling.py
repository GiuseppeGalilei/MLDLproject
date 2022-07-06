"""
Based on the implementation in https://github.com/AshwinRJ/Federated-Learning-PyTorch
"""

import numpy as np


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


def another_cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # i.i.d. selection from dataset
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


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
