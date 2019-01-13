import numpy as np
import pickle
import pandas as pd
import utils
import random


pickle_in_a = open("music_genres_dataset_aug.pkl", "rb")
dataset_a = pd.DataFrame.from_dict(pickle.load(pickle_in_a))

pickle_in = open("music_genres_dataset.pkl", "rb")
dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))


def sample(data, data_a):
    # data_a = pd.DataFrame([[row.get("data"), row.get("labels"), row.get("track_id")]  for idx, row in data_a.iterrows() if idx % 9 != 0],
    #                     columns=["data", "labels", "track_id"])[0:40000]

    groups = [data for _, data in data.groupby('track_id')]
    groups_a = [data for _, data in data_a.groupby('track_id')]

    groups_all = [[groups[I], groups_a[I]] for I in range(len(groups))]

    random.shuffle(groups_all)

    groups   = [x[0] for x in groups_all][700:]
    groups_a = [x[1] for x in groups_all][:700]

    ## checks

    while len(set(np.row_stack( np.array([row["labels"].values for row in groups])).flatten())) != 10 and len(set(np.row_stack( np.array([row["labels"].values for row in groups_a])).flatten())) != 10:
        print("Must reshuffle")
        random.shuffle(groups_all)
        groups   = [x[0] for x in groups_all][700:]
        groups_a = [x[1] for x in groups_all][:700]


    import ipdb; ipdb.set_trace()
    # random.shuffle(groups)
    # random.shuffle(groups_a)
    print("I")
    dataset = pd.concat(groups).reset_index(drop=True)
    dataset_a = pd.concat(groups_a).reset_index(drop=True)

    print("AM")
    trainBatch  = list(np.row_stack(dataset_a["data"].values))
    trainBatch  = np.array(list(map(utils.melspectrogram, trainBatch)), dtype=np.float32)

    print("SO")
    trainLabels = np.array(pd.get_dummies(dataset_a["labels"]).values, dtype=np.float32)

    testBatch   = np.row_stack(dataset["data"].values)
    testBatch  = np.array(list(map(utils.melspectrogram, testBatch)), dtype=np.float32)

    print("BORED")
    testLabels  = np.array(pd.get_dummies(dataset["labels"]).values, dtype=np.float32)

    return trainBatch, testBatch, trainLabels, testLabels


import ipdb; ipdb.set_trace()