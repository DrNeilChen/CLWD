import inspect
import torch
import importlib
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from utils.evaluate import evaluation, plot_evaluation_curves, save_predictions



class CLAMMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.tra_outs = []
        self.tra_labels = []
        self.val_outs = []
        self.val_labels = []
        self.test_outs = []
        self.test_labels = []

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, label, filename = batch

        out, A_raw, instance_dict = self([img, label])
        bagloss = self.loss_function(out, label)
        instance_loss = instance_dict["instance_loss"]
        loss = bagloss * 0.7 + 0.3 * instance_loss

        self.tra_outs.append(out.detach())
        self.tra_labels.append(label)
        self.log("tra_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, filename = batch
        out, A_raw, instance_dict = self([img, label])
        bagloss = self.loss_function(out, label)
        instance_loss = instance_dict["instance_loss"]
        loss = bagloss * 0.7 + 0.3 * instance_loss

        self.val_outs.append(out.detach())
        self.val_labels.append(label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label, filename = batch
        out, A_raw, instance_dict = self([img, label])
        bagloss = self.loss_function(out, label)
        instance_loss = instance_dict["instance_loss"]
        loss = bagloss * 0.7 + 0.3 * instance_loss

        self.test_outs.append(out.detach())
        self.test_labels.append(label)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        outs = torch.cat(self.tra_outs).cpu().numpy()
        labels = torch.cat(self.tra_labels).cpu().numpy()
        self.tra_outs.clear()
        self.tra_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "tra_acc",
            results['ACC'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        outs = torch.cat(self.val_outs).cpu().numpy()
        labels = torch.cat(self.val_labels).cpu().numpy()
        self.val_outs.clear()
        self.val_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "val_acc",
            results['ACC'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        outs = torch.cat(self.test_outs).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        self.test_outs.clear()
        self.test_labels.clear()
        results = evaluation(outs, labels)
        class_names = ["papillary","lepidic","in situ","solid","micropapillary","cribriform","acinar"]
        plot_evaluation_curves(labels, outs, class_names, save_dir=self.hparams.save_dir, kfold=self.hparams.k)
        save_predictions(outs, labels, save_dir=self.hparams.save_dir)
        e = {
            'ACC': results['ACC'],
            'AUC': results['AUC'],
            'Recall': results['Recall'],
            'Precision': results['Precision'],
            'F1': results['F1']
        }
        self.log_dict(dictionary=e, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.weight_decay is not None:
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
        )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            elif self.hparams.lr_scheduler == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lr_decay_steps,
                    eta_min=self.hparams.lr_decay_min_lr,
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == "ce":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            Model = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


from .optimizer.lookahead import Lookahead

class TranMILMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.tra_outs = []
        self.tra_labels = []
        self.val_outs = []
        self.val_labels = []
        self.test_outs = []
        self.test_labels = []

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)    

        self.tra_outs.append(out.detach())
        self.tra_labels.append(label)
        self.log("tra_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)

        self.val_outs.append(out.detach())
        self.val_labels.append(label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)

        self.test_outs.append(out.detach())
        self.test_labels.append(label)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        outs = torch.cat(self.tra_outs).cpu().numpy()
        labels = torch.cat(self.tra_labels).cpu().numpy()
        self.tra_outs.clear()
        self.tra_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "tra_acc",
            results['ACC'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        outs = torch.cat(self.val_outs).cpu().numpy()
        labels = torch.cat(self.val_labels).cpu().numpy()
        self.val_outs.clear()
        self.val_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "val_acc",
            results['ACC'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        outs = torch.cat(self.test_outs).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        self.test_outs.clear()
        self.test_labels.clear()
        results = evaluation(outs, labels)
        class_names = ["papillary","lepidic","in situ","solid","micropapillary","cribriform","acinar"]
        plot_evaluation_curves(labels, outs, class_names, save_dir=self.hparams.save_dir, kfold=self.hparams.k)
        save_predictions(outs, labels, save_dir=self.hparams.save_dir)
        
        e = {
            'ACC': results['ACC'],
            'AUC': results['AUC'],
            'Recall': results['Recall'],
            'Precision': results['Precision'],
            'F1': results['F1']
        }
        self.log_dict(dictionary=e, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.weight_decay is not None:
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        if self.hparams.optimizer == "adam" or self.hparams.optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == "lookaheadadam":
            optimizer = torch.optim.RAdam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
            optimizer = Lookahead(optimizer)
        else:
            raise ValueError("Invalid optimizer type!")

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            elif self.hparams.lr_scheduler == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lr_decay_steps,
                    eta_min=self.hparams.lr_decay_min_lr,
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == "bce":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss == "mls":
            self.loss_function = nn.MultiLabelSoftMarginLoss()
        elif loss == "ce":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            Model = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

class GraphTransformerMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.tra_outs = []
        self.tra_labels = []
        self.val_outs = []
        self.val_labels = []
        self.test_outs = []
        self.test_labels = []

    def preparefeatureLabel(
        self, batch_graph, batch_label, batch_adjs, n_features: int = 512
    ):
        batch_size = len(batch_graph)
        device = batch_graph[0].device
        label = torch.zeros(batch_size, len(batch_label[0]))
        max_node_num = 0

        for i in range(batch_size):
            label[i] = batch_label[i]
            max_node_num = max(max_node_num, batch_graph[i].shape[0])

        masks = torch.zeros(batch_size, max_node_num)
        adjs = torch.zeros(batch_size, max_node_num, max_node_num)
        batch_node_feat = torch.zeros(batch_size, max_node_num, n_features)

        for i in range(batch_size):
            cur_node_num = batch_graph[i].shape[0]
            # node attribute feature
            tmp_node_fea = batch_graph[i]
            batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

            # adjs
            adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]

            # masks
            masks[i, 0:cur_node_num] = 1
        batch_node_feat = batch_node_feat.to(device)
        label = label.to(device)
        adjs = adjs.to(device)
        masks = masks.to(device)
        return batch_node_feat, label, adjs, masks

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, adjs, label, filename = batch
        node_feat, label, adjs, masks = self.preparefeatureLabel(
            img, label, adjs, n_features=512
        )
        out, _, loss = self((node_feat, label, adjs, masks))

        self.tra_outs.append(out.detach())
        self.tra_labels.append(label)

        self.log("tra_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, adjs, label, filename = batch
        node_feat, label, adjs, masks = self.preparefeatureLabel(
            img, label, adjs, n_features=512
        )
        out, _, loss = self((node_feat, label, adjs, masks))

        self.val_outs.append(out.detach())
        self.val_labels.append(label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, adjs, label, filename = batch
        node_feat, label, adjs, masks = self.preparefeatureLabel(
            img, label, adjs, n_features=512
        )
        out, _, loss = self((node_feat, label, adjs, masks))
        
        self.test_outs.append(out.detach())
        self.test_labels.append(label)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        outs = torch.cat(self.tra_outs).cpu().numpy()
        labels = torch.cat(self.tra_labels).cpu().numpy()
        self.tra_outs.clear()
        self.tra_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "tra_acc",
            results["ACC"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        outs = torch.cat(self.val_outs).cpu().numpy()
        labels = torch.cat(self.val_labels).cpu().numpy()
        self.val_outs.clear()
        self.val_labels.clear()
        results = evaluation(outs, labels)
        self.log(
            "val_acc",
            results["ACC"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        outs = torch.cat(self.test_outs).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        self.test_outs.clear()
        self.test_labels.clear()
        results = evaluation(outs, labels)
        class_names = ["papillary","lepidic","in situ","solid","micropapillary","cribriform","acinar"]
        plot_evaluation_curves(labels, outs, class_names, save_dir=self.hparams.save_dir, kfold=self.hparams.k)
        save_predictions(outs, labels, save_dir=self.hparams.save_dir)
        e = {
            'ACC': results['ACC'],
            'AUC': results['AUC'],
            'Recall': results['Recall'],
            'Precision': results['Precision'],
            'F1': results['F1']
        }
        self.log_dict(dictionary=e, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.weight_decay is not None:
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
        )
        optimizer = Lookahead(optimizer)
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == "steplr":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            elif self.hparams.lr_scheduler == "cosinelr":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lr_decay_steps,
                    eta_min=self.hparams.lr_decay_min_lr,
                )
            elif self.hparams.lr_scheduler == "multisteplr":
                scheduler = lrs.MultiStepLR(
                    optimizer,
                    milestones=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]

    def load_model(self):
        name = self.hparams.model_name
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            Model = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
