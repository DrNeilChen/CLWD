import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ClAMDInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_workers = kwargs["num_workers"]
        self.dataset = kwargs["dataset"]
        self.batch_size = kwargs["batch_size"]
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset = self.instancialize(train="train")
            self.valset = self.instancialize(train="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.testset = self.instancialize(train="test")


    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def load_data_module(self):
        name = self.dataset
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


class GraphTransformerDInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_workers = kwargs["num_workers"]
        self.dataset = kwargs["dataset"]
        self.batch_size = kwargs["batch_size"]
        self.load_data_module()

    def collate(self, batch):
        image = [b[0] for b in batch]  # w, h
        label = [b[2] for b in batch]
        id = [b[3] for b in batch]
        adj_s = [b[1] for b in batch]
        return (image, adj_s, label, id)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset = self.instancialize(train="train")
            self.valset = self.instancialize(train="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.testset = self.instancialize(train="test")

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
            pin_memory=True
        )

    def load_data_module(self):
        name = self.dataset
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

