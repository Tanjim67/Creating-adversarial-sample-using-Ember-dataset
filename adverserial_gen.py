import math
import json
import os
from sklearn.preprocessing import StandardScaler
import pickle


def adv_examples_to_json(examples, data_dir, scaler):
    """
    reads in adversarial examples, scales it to original vectorized form, collects the raw data, and perturbs the raw
    data according to the adversarial examples
    :param examples: adversarial examples in the form they are returned by the attack
    :param data_dir: the folder where the ember dataset and its vectorized data lies
    :param scaler: the scaler with which the data was originally scaled
    :return: A list of Dictionaries, each element representing the raw data
    """
    print("extracting hashes and rescaling")
    hashes = []
    X_adv_p = []
    for x, x_pert, hash_nums, y, out_orig, out_pert in examples:
        if y == 1 and out_orig == 1 and out_pert == 0:
            hash_chars = [chr(element) for element in hash_nums]
            hash = "".join(hash_chars)
            hashes.append(hash)
            X_adv_p.append(x_pert.flatten().cpu().detach().numpy())
    X_adv_p = scaler_invert(X_adv_p, scaler=scaler)

    json_features = len(hashes) * [None]

    print("loading raw files")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    for path in raw_feature_paths:
        with open(path) as file_in:
            for line in file_in:
                if json_features.count(None) == 0:
                    break

                loads = json.loads(line)
                sha256 = loads["sha256"]
                try:
                    i = hashes.index(sha256)
                except ValueError:
                    continue

                json_features[i] = loads

    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    for path in raw_feature_paths:
        with open(path) as file_in:
            for line in file_in:
                if json_features.count(None) == 0:
                    break

                loads = json.loads(line)
                sha256 = loads["sha256"]
                try:
                    i = hashes.index(sha256)
                except ValueError:
                    continue

                json_features[i] = loads

    print("perturb raw files")
    for i in range(len(hashes)):
        json_feature = json_features[i]
        if json_feature is None:
            continue

        perturbed = X_adv_p[i]

        # perturbed_reversed = reverse(perturbed)

        header_vectorized = perturbed[626:688]
        json_feature['header']['coff']['timestamp'] = int(header_vectorized[0])

        # impossible to reverse because of hashing:
        # FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
        # FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
        # FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
        # FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
        # FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],

        json_feature['header']['optional']['major_image_version'] = int(header_vectorized[51])
        json_feature['header']['optional']['minor_image_version'] = int(header_vectorized[52])
        json_feature['header']['optional']['major_linker_version'] = int(header_vectorized[53])
        json_feature['header']['optional']['minor_linker_version'] = int(header_vectorized[54])
        json_feature['header']['optional']['major_operating_system_version'] = int(header_vectorized[55])
        json_feature['header']['optional']['minor_operating_system_version'] = int(header_vectorized[56])
        json_feature['header']['optional']['major_subsystem_version'] = int(header_vectorized[57])
        json_feature['header']['optional']['minor_subsystem_version'] = int(header_vectorized[58])
        json_feature['header']['optional']['sizeof_code'] = int(header_vectorized[59])  # 685 bigger
        json_feature['header']['optional']['sizeof_headers'] = int(header_vectorized[60])  # 686 bigger
        json_feature['header']['optional']['sizeof_heap_commit'] = int(header_vectorized[61])  # 687 bigger

        general_vectorized = perturbed[616:626]
        json_feature['general']['size'] = int(general_vectorized[0])  # 616 bigger
        json_feature['general']['vsize'] = int(general_vectorized[1])  # 617 bigger
        json_feature['general']['has_debug'] = int(general_vectorized[2])  # 618 bool
        json_feature['general']['exports'] = int(general_vectorized[3])  # 619 bigger
        json_feature['general']['imports'] = int(general_vectorized[4])  # 620 bigger
        json_feature['general']['has_relocations'] = int(general_vectorized[5])  # 621 bool
        json_feature['general']['has_resources'] = int(general_vectorized[6])  # 622 bool
        json_feature['general']['has_signature'] = int(general_vectorized[7])  # 623 bool
        json_feature['general']['has_tls'] = int(general_vectorized[8])  # 624 bool
        json_feature['general']['symbols'] = int(general_vectorized[9])  # 625 bigger

        strings_vectorized = perturbed[512:616]
        json_feature['strings']['numstrings'] = int(strings_vectorized[0])
        json_feature['strings']['avlength'] = float(
            strings_vectorized[1])  # TODO not precisely the same as original value
        json_feature['strings']['printables'] = int(strings_vectorized[2])

        hist_multiplicator = float(json_feature['strings']['printables']) if json_feature['strings'][
                                                                                 'printables'] > 0 else 1.0
        printabledist = []
        for i in range(96):
            f_value = strings_vectorized[3 + i] * hist_multiplicator
            rest = f_value - math.floor(f_value)
            if rest >= 0.5:
                f_value = math.ceil(f_value)
            else:
                f_value = int(f_value)
            printabledist.append(int(f_value))
        json_feature['strings']['printabledist'] = printabledist

        json_feature['strings']['entropy'] = float(
            strings_vectorized[99])  # TODO not precisely the same as original value
        json_feature['strings']['paths'] = int(strings_vectorized[100])
        json_feature['strings']['urls'] = int(strings_vectorized[101])
        json_feature['strings']['registry'] = int(strings_vectorized[102])
        json_feature['strings']['MZ'] = int(strings_vectorized[103])

        json_features[i] = json_feature

    return json_features


def save_adv_raw(save_dir, json_features):
    """
    Saves a list of Dictionaries, each element representing the raw data into a jsonl file, each line will be one sample
    :param save_dir: the directory where to save the data
    :param json_features:  the list with raw data
    :return: None
    """
    save_dir = save_dir + "adv_examples.jsonl"
    json.dump(json_features[0], open(save_dir, "w"))
    file = open(save_dir, "a")
    for json_feature in json_features[1:]:
        file.write("\n")
        json.dump(json_feature, file)
    file.close()


def scaler_invert(X_train, scaler: StandardScaler = None, scaler_dir: str = None):
    """
    Inverts the application of the scaler for a given numpy array. You have to either set scaler or the directory where
    the scaler is saved
    :param X_train: numpy array on which the scaler application will be inverted
    :param scaler: the scaler (optional)
    :param scaler_dir: the directory where the scaler is saved
    :return: Altered data (X_train, numpy array)
    """
    if scaler is None:
        if scaler_dir is not None:
            scaler = pickle.load(open(scaler_dir + 'scaler.pkl', 'rb'))
        else:
            raise Exception("you should declare the scaler or a path where the scaler is saved ")
    return scaler.inverse_transform(X_train)
