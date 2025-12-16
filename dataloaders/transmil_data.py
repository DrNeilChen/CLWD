import torch
import os
import numpy as np
import torch.utils.data as data
import pandas as pd



class TransmilData(data.Dataset):
    def __init__(
        self,
        data_dir="",
        dataset="",
        label_file="",
        train="train",
        k=0,
        fold_num=0,
        seed=3407,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.check_files()

    def check_files(self):

        fl_train = (
            pd.read_csv(os.path.join(self.label_file, f"train{self.k}.csv"))
            .sample(frac=1)
        )
        fl_val = (
            pd.read_csv(os.path.join(self.label_file, f"val{self.k}.csv"))
            .sample(frac=1)
        )
        fl_test = (
            pd.read_csv(os.path.join(self.label_file, "test.csv"))
            .sample(frac=1)
        )
        if self.train == "train":
            self.path_list = fl_train
        elif self.train == "val":
            self.path_list = fl_val
        else:
            self.path_list = fl_test

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.path_list.iloc[idx,0])
        g_name = os.path.splitext(os.path.basename(path))[0]
        label = self.path_list.iloc[idx,1:].values.astype(np.int8)
        features = torch.load(
            os.path.join(self.data_dir, g_name, "features.pt"),
            map_location="cpu",
        )

        return features, torch.FloatTensor(label), g_name
