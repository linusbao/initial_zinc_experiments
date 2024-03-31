# import os
import numpy as np
import random
# import copy

import torch
from torch.optim import Adam
# import torch_geometric.seed
# from torch_geometric.loader import DataLoader
# from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

class GPS_experiment():
    
    def __init__(self, model, data, hyperparam_dict, n_runs=4, print_train_loss_every=20):
        #object parameters
        self.model = model
        self.data = data

        self.lr = hyperparam_dict['lr']
        self.lr_factor = hyperparam_dict['lr_factor']
        self.lr_patience = hyperparam_dict['lr_patience']
        self.n_epochs = hyperparam_dict['n_epochs']
        self.min_lr = hyperparam_dict['min_lr']

        self.n_runs = n_runs

        #misc
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Running on device", self.device)

        #settings
        self.print_train_loss_every = print_train_loss_every

        #stored data from training and testing
        #self.og_model_state = copy.deepcopy(model.state_dict()) #QUESTION: will we ever get different results if we start eachh run from the same exact state_dict (do we need to reintialize model each run to get varuiance)?
        self.record = []
        self.result = {'avg' : -1, "plus_minus" : -1}

    def run(self):
        m = self.model
        D_train = self.data['train']
        D_val = self.data['validation']
        D_test = self.data['test']

        MAEs = []
        for i in range(self.n_runs):
            #m.load_state_dict(og_model_state)
            m.apply(self.weight_reset)
            train_outputs = self.train(m, D_train, D_val, D_test)
            self.record.append(train_outputs)
            MAEs.append(train_outputs['best_test_perf'])

        self.result['avg'] = sum(MAEs) / len(MAEs)
        self.result['plus_minus'] = torch.var(torch.tensor(MAEs)).item()

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # def plot(#TODO options):

    #     if len(self.record) == 0:
    #         print('Run first')
    #     else:
    #         #TODO


    def training_epoch(self, m, D, loss_fn, opt):
        m.train()
        opt.zero_grad()

        losses = []
        for batch in D:

            batch = batch.to(self.device)
            pred = m(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.to(torch.float32).view(pred.shape)
            loss = loss_fn(pred, target)

            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().item())
        
        return sum(losses) / len(losses)

    def eval(self, m, D):
        m.eval()

        loss_fn = torch.nn.L1Loss()
        y_true = []
        y_pred = []
        losses = []
        
        for batch in D:
            # Cast features to double precision if that is used
            if torch.get_default_dtype() == torch.float64:
                for dim in range(batch.dimension + 1):
                    batch.cochains[dim].x = batch.cochains[dim].x.double()
                    assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

            batch = batch.to(self.device)
            with torch.no_grad():
                pred = m(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                targets = batch.y.to(torch.float32).view(pred.shape)
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                loss = loss_fn(pred, targets)
                losses.append(loss.detach().cpu().item())

            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
        y_pred = torch.cat(y_pred, dim=0).numpy()
        
        mae = mean_absolute_error(y_true, y_pred)
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan

        return mae, mean_loss

    def train(self, m, D_train, D_val, D_test):
        m.to(self.device)

        loss_fn = torch.nn.L1Loss()
        opt = Adam(m.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=self.lr_factor, patience=self.lr_patience, verbose=True)
        
        best_val_epoch = 0
        training_avg_losses = []
        train_curve = []
        val_curve = []
        test_curve = []
        epoch_val_losses = []
        epoch_test_losses = []

        for epoch in range(self.n_epochs):
            avg_train_loss = self.training_epoch(m, D_train, loss_fn, opt)
            training_avg_losses.append(avg_train_loss)
            if epoch % self.print_train_loss_every:
                print(f'Avg training loss at epoch {epoch} is {avg_train_loss}')

            train_perf, _ = self.eval(m, D_train)
            train_curve.append(train_perf)
            val_perf, epoch_val_loss = self.eval(m, D_val)
            val_curve.append(val_perf)
            epoch_val_losses.append(epoch_val_loss)
            test_perf, epoch_test_loss = self.eval(m, D_test)
            test_curve.append(test_perf)
            epoch_test_losses.append(epoch_test_loss)

            scheduler.step(val_perf)

            if opt.param_groups[0]['lr'] < self.min_lr:
                print("\n!! The minimum learning rate has been reached.")
                break

        best_val_epoch = np.argmin(np.array(val_curve))
        total_n_params = sum(param.numel() for param in m.parameters())

        return {"training_avg_losses": training_avg_losses, 'epoch_val_losses': epoch_val_losses, 'epoch_test_losses': epoch_test_losses, 'train_curve': train_curve, "val_curve" : val_curve, "test_curve" : test_curve, "best_test_perf": test_curve[best_val_epoch], "best_val_epoch": best_val_epoch, "total_n_params" : total_n_params}
