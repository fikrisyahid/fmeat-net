# FMEAT-Net: An Efficient CNN Architecture for Image-Based Identification of Beef, Pork, and Adulterated Meat

## Table of contents
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Steps to run the code](#steps-to-run-the-code)
- [Steps to run deployment](#steps-to-run-deployment)
- [Contact](#contact)

## Description
This repository contains the code and resources for the paper `FMEAT-Net: An Efficient CNN Architecture for Image-Based Identification of Beef, Pork, and Adulterated Meat`.

![fmeat-net architecture](https://github.com/user-attachments/assets/9b58cabf-bee0-44ab-adc3-f12946b07850)
FMEAT-Net architecture visualization

## Prerequisites
You need to have the following libraries installed:
- PyTorch: `torch torchvision torchaudio`
- torchsummary: `torchsummary`
- pandas: `pandas openpyxl`
- numpy: `numpy`
- seaborn: `seaborn`
- matplotlib: `matplotlib`
- tqdm: `tqdm`
- pynvml: `pynvml`
- scikit-learn: `scikit-learn`
- flask: `flask`

> We used `Python 3.12.9` and we recommend you to use the same version, since at the time of writing this README, `Python > 3.12` is still buggy.

## Steps to run the code
1. Clone the repository: `git clone https://github.com/fikrisyahid/fmeat-net.git`
2. Change directory: `cd fmeat-net`
3. Run `python run_experiments.py` to start the whole combination training and evaluation process.

Notes:
- You can customize the configs in `config.py` file.
- You can run the combination individually by running `python main.py`. You can see the help by running `python main.py -h`.

## Steps to run deployment
1. Make sure you have flask installed
2. Run `flask --app deployment/app run`
3. Open your browser and go to `http://127.0.0.1:5000/`

## Contact
If you have any questions or suggestions, please feel free to contact me at `fikrisyahid@apps.ipb.ac.id`
