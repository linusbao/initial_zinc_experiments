import torch
from torch.utils.data import random_split
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from GPS_experiment import GPS_experiment
from get_zinc_data_gps import load_zinc_homcount_dataset, load_zinc_subcount_dataset, load_zinc_subhom_dataset, load_zinc_dataset
from GPS_model import GPS

def main():
    #HYPERPARAMS
    SUBSET = None #set to decimal value less than 1 to obtain percentage of zinc12K data
    BATCH_SIZE = 32
    model_hyperparam_dict = {
        'lr' : 0.001,
        'n_epochs' : 2000, #should be 2000, just wanna test quickly
        'lr_factor' : 0.5,
        'lr_patience' : 20,
        'min_lr' : 0.00001
    }

    root = '/data/coml-graphml/kell7068/hombasis-gnn/hombasis-bench/data/zinc-data' #CHANGE TO PATH OF "zinc-data" folder
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'using device {device}')

    PE_datasets = {
        'RandomWalk': load_zinc_dataset('ZINC', root, transform=T.AddRandomWalkPE(walk_length=20, attr_name='pe')),
        'AnchoredHom': load_zinc_homcount_dataset('ZINC', ['zinc_with_anchored_homs_c78_full.json'], [], root)
                }

    experiments = {}
    for PE_type, dataset in PE_datasets.items():

        train_dataset, val_dataset, test_dataset, pe_init_dim = dataset

        #sets inital PE dim to 20 for the case where the PE is random walk
        if pe_init_dim == None:
            pe_init_dim = 20

        #[optional] take subset of data
        if SUBSET != None:
            train_dataset, _ = random_split(train_dataset, [SUBSET, 1-SUBSET])
            val_dataset, _ = random_split(val_dataset, [SUBSET, 1-SUBSET])
            test_dataset, _ = random_split(test_dataset, [SUBSET, 1-SUBSET])
            
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        data = {'train' : train_loader, 'validation': val_loader, 'test' : test_loader}

        model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type='multihead', attn_kwargs={'dropout': 0.5}, pe_init_dim=pe_init_dim).to(device)
        
        experiments[PE_type] = GPS_experiment(model, data, model_hyperparam_dict)

    for PE_type, exp in experiments.items():

        print(f'Running {PE_type} PE experiment')
        exp.run()
        print(f'Final avg MAE = {exp.result['avg']}')
        print(f'Final MAE variance = {exp.result['plus_minus']}')

    for PE_type, exp in experiments.items():

        print("(re)Printing final results from all experiments")
        print(f'Final MAE for {PE_type} PE: {exp.result['avg']} +- {exp.result['plus_minus']}')
        print('Done!')


if __name__ == "__main__":
    main()