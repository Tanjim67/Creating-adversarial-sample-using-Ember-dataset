import numpy as np
import tqdm
import torch
import pandas as pd
pd.set_option('display.max_colwidth', None)

def norm(order, examples):
    norms_per_epsilon = []
    for adv_example in examples:
        accu = 0
        for data_orig, data_pert, _, _, _, _ in adv_example:
            delta = data_orig[0]-data_pert[0]
            #L0 not supported by torch.linalg.norm, own implementation in if statement
            if order==0:
                accu += (delta != 0).sum().item()
            else:
                accu += torch.linalg.norm(delta, ord=order).item()
        norms_per_epsilon.append(accu/len(adv_example))
    return norms_per_epsilon

def norms(examples):
    ret = []
    orders = [0,1,2,np.inf]

    for order in tqdm.auto.tqdm(orders):
        ret.append(norm(order, examples))
    return ret

def calc_percentage(x,y):
    if y == 0:
        return 0
    return np.round(x / y * 100,2)

def calculate_shares(adv_ex):         
    benign = [0,0,0,0,0,0,0]
    malicious = [0,0,0,0,0,0,0]
    for _ , _, _, target, orig, pert in adv_ex:
        if target.item() == 0:
            benign[0] = benign[0] + 1
            if orig.item() == 0:
                benign[1] = benign[1] + 1
                if pert.item() == 0:
                    benign[3] = benign[3] + 1
                elif pert.item() == 1:
                    malicious[3] = malicious[3] + 1
            elif orig.item() == 1:
                malicious[1] = malicious[1] + 1
                if pert.item() == 0:
                    benign[4] = benign[4] + 1
                elif pert.item() == 1:
                    malicious[4] = malicious[4] + 1
        elif target.item() == 1:
            malicious[0] = malicious[0] + 1
            if orig.item() == 0:
                benign[2] = benign[2] + 1
                if pert.item() == 0:
                    benign[5] = benign[5] + 1
                elif pert.item() == 1:
                    malicious[5] = malicious[5] + 1
            elif orig.item() == 1:
                malicious[2] = malicious[2] + 1
                if pert.item() == 0:
                    benign[6] = benign[6] + 1
                elif pert.item() == 1:
                    malicious[6] = malicious[6] + 1
        percentages = [
            calc_percentage(benign[1], benign[0]),
            calc_percentage(malicious[1], benign[0]),
            calc_percentage(benign[2], malicious[0]),
            calc_percentage(malicious[2], malicious[0]),
            calc_percentage(benign[3], benign[1]),
            calc_percentage(malicious[3], benign[1]),
            calc_percentage(benign[4], malicious[1]),
            calc_percentage(malicious[4], malicious[1]),
            calc_percentage(benign[5], benign[2]),
            calc_percentage(malicious[5], benign[2]),
            calc_percentage(benign[6], malicious[2]),
            calc_percentage(malicious[6], malicious[2])
        ]
    return benign, malicious, percentages

def create_dfs_swaps(benign, malicious, percentages):
    d = {
        'Meaning': ["original benign","original benign & predicted benign","original benign & predicted benign & attack predicted benign","original benign & predicted benign & attack predicted malicious","original benign & predicted malicious","original benign & predicted malicious & attack predicted benign","original benign & predicted malicious & attack predicted malicious","original malicious","original malicious & predicted benign","original malicious & predicted benign & attack predicted benign","original malicious & predicted benign & attack predicted malicious","original malicious & predicted malicious","original malicious & predicted malicious & attack predicted benign","original malicious & predicted malicious & attack predicted malicious"],
        'Field in list': ["benign[0]","benign[1]","benign[3]","malicious[3]","malicious[1]","benign[4]","malicious[4]","malicious[0]","benign[2]","benign[5]","malicious[5]","malicious[2]","benign[6]","malicious[6]"],
        'no of samples': [benign[0],benign[1],benign[3],malicious[3],malicious[1],benign[4],malicious[4],malicious[0],benign[2],benign[5],malicious[5],malicious[2],benign[6],malicious[6]]
    }
    df_absolute = pd.DataFrame(data=d)
    d = {
        'Proportion of samples which are' : ["predicted benign", "predicted malicious","predicted benign", "predicted malicious","attack predicted benign","attack predicted malicious","attack predicted benign","attack predicted malicious","attack predicted benign","attack predicted malicious","attack predicted benign","attack predicted malicious"],
        'among samples which are' : ["original benign","original benign","original malicious","original malicious","original benign & predicted benign","original benign & predicted benign","original benign & predicted malicious","original benign & predicted malicious","original malicious & predicted benign","original malicious & predicted benign","original malicious & predicted malicious","original malicious & predicted malicious"],
        'percentage': percentages
    }
    df_percentage = pd.DataFrame(data=d)
    return df_absolute, df_percentage
