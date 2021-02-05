#Content


This is the repository for the final group project for the lecture [Machine Learning in Cyber Security](https://cms.cispa.saarland/mlcysec2021/) at CISPA / Saarland University. 

It contains an experiment about malware classification on the [EMBER dataset](https://github.com/elastic/ember) with Neural Nets and it vulnerability to evasion attacks, especially FGSM (Fast Gradient Signed Method). We also propose adversarial training as a defense mechanism against such attacks.

#Structure

📂mlcysec_final_project
 ┣ 📂 [adv_samples]
 ┃ ┣ 📜 adv_examples.jsonl
 ┃ ┣ 📜 [perturbed_example_pretty.json](adv_samples/perturbed_example_pretty.json)
 ┃ ┣ 📜 selected_samples_original.jsonl
 ┃ ┗ 📜 selected_samples_perturbed.jsonl
 ┣ 📂 ember_github
 ┣ 📂 model
 ┃ ┣ 📜 EmberNet2.pth
 ┃ ┣ 📜 EmberNet2_hist.pth
 ┃ ┣ 📜 EmberNetRobust.pth
 ┃ ┣ 📜 EmberNetRobust_hist.pth
 ┃ ┗ 📜 scaler.pkl
 ┣ 📜 README.md
 ┣ 📦 [adverserial_gen.py](adverserial_gen.py)
 ┣ 📦 [ember_net.py](ember_net.py)
 ┣ 📜 [ember_nn.ipynb](ember_nn.ipynb)
 ┣ 📜 [ember_nn_robust.ipynb](ember_nn_robust.ipynb)
 ┣ 📦 [evaluation.py](evaluation.py)
 ┣ 📦 [plots.py](plots.py)
 ┗ 📦 [preprocessing.py](preprocessing.py)

 In the folder ```adv_examples``` you will find several samples from our adverserial sample set. The folder ```ember_github``` is a fork of the [EMBER repository](https://github.com/elastic/ember). In the folder ```model``` you will find our pretrained models, including training history.

 The notebooks ```ember_nn.ipynb```,```ember_nn_robust.ipynb ``` contain our training process and pipeline with results. The first notebook for training FGSM attack and adversarial sample gerneration, the second one for adversarial training and FGSM attack. The module ```ember_net.py``` contains our Neural Network definition, ```adverserial_gen.py```, ```evaluation.py```, ```plots.py``` and ```preprocessing.py``` are further modules we created to to modularize our project.

#Usage

To use the provided notebooks and models, You should first download the ember dataset from their [repository](https://github.com/elastic/ember). We used the feature version 2018. To work seamlessly, the dataset should be downloaded into a folder called ```ember2018``` inside the root folder of this project. You can also specify any other folder if necessary. 

You then can open one of the notebooks, which are self-explanatory. It is important to first vectorize the dataset and collect the hashes of each sample. Both needs only to be done once. You can do so be setting the parameter of the data preprocessing accordingly.


Please note that this project most likely will not work on windows.