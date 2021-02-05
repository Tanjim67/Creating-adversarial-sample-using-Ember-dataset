import json
import os
import pickle
from typing import Tuple

import altair as alt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import ember_github.ember as ember

_ = alt.renderers.enable('default')


def vectorize_raw(data_dir: str):
    """
    Vectorizes the Ember Dataset using provided function
    :param data_dir: the folder where the ember dataset was unzipped
    :return: None
    """
    ember.create_vectorized_features(data_dir)
    _ = ember.create_metadata(data_dir)


def show_metadata(data_dir: str):
    """
    Shows some information about the ember dataset. Needs altair to be configured correctly
    :param data_dir: the folder where the ember dataset and its vectorized data lies
    :return: None
    """
    emberdf = ember.read_metadata(data_dir)

    plotdf = emberdf.copy()
    gbdf = plotdf.groupby(["label", "subset"]).count().reset_index()
    chart = alt.Chart(gbdf).mark_bar().encode(
        alt.X('subset:O', axis=alt.Axis(title='Subset')),
        alt.Y('sum(sha256):Q', axis=alt.Axis(title='Number of samples')),
        alt.Color('label:N', scale=alt.Scale(range=["#00b300", "#3333ff", "#ff3333"]),
                  legend=alt.Legend(values=["unlabeled", "benign", "malicious"]))
    )
    chart.show()

    plotdf = emberdf.copy()
    plotdf.loc[plotdf["appeared"] < "2018-01", "appeared"] = " <2018"
    gbdf = plotdf.groupby(["appeared", "label"]).count().reset_index()
    chart = alt.Chart(gbdf).mark_bar().encode(
        alt.X('appeared:O', axis=alt.Axis(title='Month appeared')),
        alt.Y('sum(sha256):Q', axis=alt.Axis(title='Number of samples')),
        alt.Color('label:N', scale=alt.Scale(range=["#00b300", "#3333ff", "#ff3333"]),
                  legend=alt.Legend(values=["unlabeled", "benign", "malicious"]))
    )
    chart.show()


def collect_hashes_from_raw(data_dir: str):
    """
    Collects hashes from raw data and saves to files in data_dir.
    :param data_dir: the folder where the ember dataset and its vectorized data lies
    :return: None
    """

    print("hashes training set")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    hashes_train = []
    for path in raw_feature_paths:
        with open(path) as file_in:
            for line in file_in:
                hash_chars = list(json.loads(line)['sha256'])
                hash_chars_numbers = [ord(element) for element in hash_chars]
                hashes_train.append(hash_chars_numbers)

    print("hashes test set")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    hashes_test = []
    for path in raw_feature_paths:
        with open(path) as file_in:
            for line in file_in:
                hash_chars = list(json.loads(line)['sha256'])
                hash_chars_numbers = [ord(element) for element in hash_chars]
                hashes_test.append(hash_chars_numbers)

    np.save(data_dir + '/hashes_train.npy', np.array(hashes_train, dtype=np.int64))
    np.save(data_dir + '/hashes_test.npy', np.array(hashes_test, dtype=np.int64))


def load_hashes(data_dir: str):
    """
    Loads saved hashes for all samples
    :param data_dir: the folder where the ember dataset and its vectorized data lies
    :return: hashes_train, hashes_test
    """
    hashes_train = np.load(data_dir + '/hashes_train.npy')
    hashes_test = np.load(data_dir + '/hashes_test.npy')
    return hashes_train, hashes_test


def import_data(data_dir: str):
    """
    Reads the vectorized data and returns four arrays with the dataset
    :param data_dir: the folder where the ember dataset and its vectorized data lies
    :return: X_train (train features), y_train (train targets), X_test (testing features), y_test (testing targets)
    """
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
    return X_train, y_train, X_test, y_test


def scale_data(X_train, y_train, X_test, y_test, scaler_dir: str):
    """
    Applies the scikit standard scaler to the data and changes dimension, so the data can be used with convolution.
    :param X_train: training features
    :param y_train: training targets
    :param X_test: testing features
    :param y_test: testing targets
    :param scaler_dir: directory where to save the standard scaler for later use
    :return: Altered data (X_train, y_train, X_test, y_test, all numpy arrays) and the scaler
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    pickle.dump(scaler, open(scaler_dir + 'scaler.pkl', 'wb'))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    return X_train, y_train, X_test, y_test, scaler


def filter_unlabeled(X_train, y_train, hashes_train):
    """
    Removes all samples from the dataset where there is no label. Our approach does not need these samples.
    :param X_train: training features
    :param y_train: training targets / labels
    :param hashes_train: hashes of training samples
    :return: Altered data (X_train, y_train, hashes, all numpy array)
    """

    return X_train[y_train != -1], y_train[y_train != -1], hashes_train[y_train != -1]


def load_data(X_train: np.ndarray, y_train: np.ndarray, hashes_train,
              X_test: np.ndarray, y_test: np.ndarray, hashes_test, batch_size: int = 100,
              validation_fragment: float = 0.1, adverserial_fragment: float = 0.001, debug_fragment: float = None
              ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Splits the data according to parameters and creates the dataloaders for training.
    :param X_train: training features
    :param y_train: training targets
    :param hashes_train: hashes of all samples in X_train in the same order.
    :param X_test: testing features
    :param y_test: testing targets
    :param hashes_test: hashes of all samples in X_train in the same order.
    :param batch_size: nr of samples for each batch
    :param validation_fragment: fragment (btw. 0 and 1) of samples used for validation
    :param adverserial_fragment: fragment (btw. 0 and 1) of samples used to generate adversarial samples
    :param debug_fragment: fragment (btw. 0 and 1) of samples to be used. For testing / debugging purposes
    :return: the four DataLoaders: train_loader, val_loader, adv_loader, test_loader
    """
    train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(hashes_train))
    test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(hashes_test))
    print("Reading {} samples from training set \nReading {} samples from test set".format(len(train), len(test)))

    if debug_fragment is not None:
        keep_train = int(len(train) * debug_fragment)
        remove_train = len(train) - keep_train
        keep_test = int(len(test) * debug_fragment)
        remove_test = len(test) - keep_test
        train, _ = torch.utils.data.random_split(train, [keep_train, remove_train])
        test, _ = torch.utils.data.random_split(test, [keep_test, remove_test])

    val_nr = int(len(train) * validation_fragment)
    adv_nr = int(len(train) * adverserial_fragment)
    train_nr = len(train) - val_nr - adv_nr

    train, val, adv = torch.utils.data.random_split(train, [train_nr, val_nr, adv_nr])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)  # part of data preprocessing
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)  #
    adv_loader = DataLoader(adv, batch_size=batch_size, shuffle=False, num_workers=1)  #
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)  # part of data preprocessing
    print("==========================================================================================",
          "\nSamples in training set: {}\nSamples in validation set: {}\nSamples in adverserial set: {}\nSamples in test set: {}"
          .format(len(train), len(val), len(adv), len(test)),
          "\n==========================================================================================")

    return train_loader, val_loader, adv_loader, test_loader


def pipeline(data_dir: str, scaler_dir: str, batch_size: int = 100,
             validation_fragment: float = 0.1, adverserial_fragment: float = 0.001, debug_fragment: float = None,
             vectorize: bool = False, collect_hashes: bool = False
             ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Chains all preprocessing together.
    :param data_dir: the folder where the ember dataset was unzipped
    :param scaler_dir: the folder where the scaler should be saved
    :param batch_size: nr of samples for each batch
    :param validation_fragment: fragment (btw. 0 and 1) of samples used for validation
    :param adverserial_fragment: fragment (btw. 0 and 1) of samples used to generate adversarial samples
    :param debug_fragment: fragment (btw. 0 and 1) of samples to be used. For testing / debugging purposes
    :param vectorize: Boolean whether or not data needs to be vectorized before loaded. Takes very long time to execute!
    :param collect_hashes: Boolean whether or not hashes need to be collectes from metadate or can be loaded from data_dir. Takes some time to execute.
    :return: the four DataLoaders: train_loader, val_loader, adv_loader, test_loader and the scaler
    """
    if vectorize:
        vectorize_raw(data_dir)
    if collect_hashes:
        collect_hashes_from_raw(data_dir)
    hashes_train, hashes_test = load_hashes(data_dir)
    X_train, y_train, X_test, y_test = import_data(data_dir)
    X_train, y_train, hashes_train = filter_unlabeled(X_train, y_train, hashes_train)
    X_train, y_train, X_test, y_test, scaler = scale_data(X_train, y_train, X_test, y_test, scaler_dir)
    train_loader, val_loader, adv_loader, test_loader = load_data(X_train, y_train, hashes_train,
                                                                  X_test, y_test, hashes_test, batch_size,
                                                                  validation_fragment, adverserial_fragment,
                                                                  debug_fragment)
    return train_loader, val_loader, adv_loader, test_loader, scaler
