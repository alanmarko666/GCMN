import torch_geometric
import torch_scatter
from tqdm import tqdm
import torch
import random
from config import config
from ogb.graphproppred import Evaluator
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
import pipeline.metrics as metrics
import wandb
from torchmetrics.functional import mean_absolute_error
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.count_params import count_params
from pipeline.metrics import binary_MAE_loss, binary_MSE_loss, binary_int_precision

import os

class Pipeline:
    def __init__(self,network,train_dataset,validation_dataset,test_datasets,batch_size=32,lr=0.001):
        self.device = config.device
        self.batch_size=batch_size
        self.lr=lr
        self.criterion=config.criterion
        self.network=network
        self.train_dataset=train_dataset
        self.test_datasets=test_datasets
        self.validation_dataset=validation_dataset
        self.network = self.network.to(self.device)
        self.molecules_evaluator = Evaluator(name = 'ogbg-molhiv')
        if config.MOLECULES:
            self.aucm_criterion = AUCMLoss().to(self.device)
        
        os.environ["WANDB_API_KEY"] = "8523288184de2d3d62b4ab6590c0cf60b918da8c"
        if not config["wandb"]:
            os.environ["WANDB_MODE"] = "disabled"
        wandb.init(project="hldnn", entity="hldnn", config=config, reinit=True)
        #wandb.run.log_code(".",exclude_fn=lambda path: ("wandb" in path or "dataset/" in path) or "data" in path)
        

    def evaluate_with(self,dataset,name, epoch):
        self.network.eval()
        with torch.no_grad():
            testloader = torch_geometric.data.DataLoader(dataset, config.EVAL_BATCH_SIZE, shuffle=False,num_workers=config.NUM_WORKERS)

            temp_sum_test_loss = 0
            temp_mae_sum=0

            y_true = []
            y_pred = []

            y_true_full_tensor=torch.ones(0,device=config.device)
            y_pred_full_tensor=torch.ones(0,device=config.device)

            for test_batch in testloader:
                # Send Batch to Device
                test_batch.to(self.device)
                
                test_batch=self.network(test_batch)

                # Forward and Loss
                y_true_tensor=test_batch.y.float()
                y_pred_tensor=test_batch.y_computed.reshape(test_batch.y.size())

                y_true_full_tensor=torch.cat([y_true_full_tensor,torch.flatten(y_true_tensor)])
                y_pred_full_tensor=torch.cat([y_pred_full_tensor,torch.flatten(y_pred_tensor)])

                temp_sum_test_loss += self.criterion(y_pred_tensor, y_true_tensor).detach().item()
                temp_mae_sum += mean_absolute_error(y_pred_tensor, y_true_tensor).detach().item()
               
                pred=test_batch.y_computed.reshape(test_batch.y.size())
                y_true.append(test_batch.y.detach())
                y_pred.append(pred.detach())


            temp_avg_test_loss = temp_sum_test_loss / len(testloader)
            temp_mae_avg = temp_mae_sum/len(testloader)

            y_true = torch.cat(y_true,dim=0)
            y_pred = torch.cat(y_pred,dim=0)
            y_true=y_true if y_true.dim()==2 else torch.unsqueeze(y_true,dim=1)
            y_pred=y_pred if y_pred.dim()==2 else torch.unsqueeze(y_pred,dim=1)

            for metric in config.METRICS:
                value = 0
                metric_name = {"AUCM":"rocauc", "LOSS": "loss", "AVERAGE_PRECISION": "AP", "MAE":"MAE",
                                "INT_PRECISION": "Integer precision", "BIN_MAE_LOSS":"Binary MAE", "BIN_MSE_LOSS":"Binary MSE",
                                "BIN_INT_PRECISION": "Binary integer precision"}[metric]

                if metric == "AUCM":
                    input_dict = {"y_true": y_true,"y_pred": y_pred}
                    value = self.molecules_evaluator.eval(input_dict)
                elif metric == "LOSS":
                    value=temp_avg_test_loss
                elif metric == "AVERAGE_PRECISION":
                    #print(y_pred[0:3])
                    value = metrics.eval_ap(y_true.cpu().numpy(),y_pred.cpu().numpy())
                elif metric == "MAE":
                    value = temp_mae_avg
                elif metric == "INT_PRECISION":
                    value = (torch.sum(y_true == torch.round(y_pred)) / y_true.flatten().shape[0]).item()
                elif metric == "BIN_MAE_LOSS":
                    value = binary_MAE_loss(y_pred, y_true).item()
                elif metric == "BIN_MSE_LOSS":
                    value = binary_MSE_loss(y_pred, y_true).item()
                elif metric == "BIN_INT_PRECISION":
                    value = binary_int_precision(y_pred, y_true).item()
                

                print(name,metric_name, ":", value)
                wandb.log({(name+"-"+metric_name): value}, step=epoch)
            
            if config.RANDOM_ROOT:
                split_size=config.REPEAT_COUNT
                y_pred_stacked=y_pred_full_tensor.view(split_size,(y_true.shape[0]//split_size),y_true.shape[1])
                y_pred_averaged=torch.mean(y_pred_stacked,dim=0)
                y_true_no_duplicates=y_true_full_tensor[0:y_pred_full_tensor.shape[0]//split_size].view((y_true.shape[0]//split_size),y_true.shape[1])
                print("Ensemble loss: ",self.criterion(y_pred_averaged,y_true_no_duplicates))
                print("Ensemble MAE: ",mean_absolute_error(y_pred_averaged,y_true_no_duplicates))
                if "AUCM" in config.METRICS:
                    y_true_no_duplicates=y_true_no_duplicates if y_true_no_duplicates.dim()==2 else torch.unsqueeze(y_true_no_duplicates,dim=1)
                    y_pred_averaged=y_pred_averaged if y_pred_averaged.dim()==2 else torch.unsqueeze(y_pred_averaged,dim=1)
                    print("Ensemble AUCM: ",self.molecules_evaluator.eval({"y_true": y_true_no_duplicates,"y_pred": y_pred_averaged}))
                #print(y_pred_stacked)
                #print(y_true_no_duplicates)
            
            return temp_avg_test_loss


    def evaluate(self, epoch):
        valid_loss=self.evaluate_with(self.validation_dataset,"Validation", epoch)
        for i, td in enumerate(self.test_datasets):
            if epoch % config.TEST_EVAL_FREQ == 0:
                self.evaluate_with(td,"Test{}".format(i+1), epoch)
        return valid_loss

    def train(self):
        print(self.network)
        print("Num params:", count_params(self.network))
        print(config)

        train_dataset=self.train_dataset

        print("Training started")
        torch.manual_seed(config.SEED)
        random.seed(config.SEED)
        data_loader = torch_geometric.loader.DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=config.NUM_WORKERS,pin_memory=False)
        print("Data loader created")
        

        
        if config.MOLECULES:
            optimizer=PESG(self.network,loss_fn=self.aucm_criterion,lr=0.1,momentum=0.9,margin=1.0,epoch_decay = 0.003, weight_decay = 0.0001)
        else:
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr)
        train_loss = []
        test_loss = []

        if config.SCHEDULER_LAMBDA is not None:
            scheduler=LambdaLR(optimizer,lr_lambda=config.SCHEDULER_LAMBDA)
        
        if config.SCHEDULER_PLATEAU:
            scheduler=ReduceLROnPlateau(optimizer,'min',0.5,min_lr=1e-5,patience=config.PATIENCE,verbose=True)

        for epoch in range(config.epoch_num):
            sum_loss = 0
            sum_abs=0
            reg_loss_sum = 0
            for step, data in tqdm(enumerate(data_loader)):
                self.network.train()
                optimizer.zero_grad()
                data = data.to(self.device)
                data = self.network(data)

                pred=data.y_computed.reshape(data.y.size())

                if config.MOLECULES:
                    is_labeled = data.y == data.y
                    loss =  self.aucm_criterion(pred.to(torch.float32)[is_labeled].reshape(-1, 1),
                                  data.y.to(torch.float32)[is_labeled].reshape(-1, 1))
                else:
                    loss = self.criterion(pred,data.y.float())

                sum_loss  += loss.detach().item()
                sum_abs +=  mean_absolute_error(pred, data.y.float()).detach().item()

                if config.REGUL_LOSS_BETA is not None:
                    reg_loss=self.network.regularizing_loss(data)
                    reg_loss_sum+=reg_loss.detach().item()
                    loss=torch.add(loss,reg_loss,alpha=config.REGUL_LOSS_BETA)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2)
                optimizer.step()

                #Logging
                train_loss.append(loss.detach().item())

                if step%500 == 499:
                    print("Partial train loss: ",train_loss[-1])
                    print("Partial train MAE: ",mean_absolute_error(pred, data.y.float()).detach().item())
                    self.evaluate(epoch)

            sum_loss/=len(data_loader)
            sum_abs/=len(data_loader)

            # Print Information
            print('-'*20)
            print('Epoch', epoch)
            print("Train loss: ",sum_loss)
            print("Train mae: ",sum_abs)
            wandb.log({"Train metric":sum_loss}, step=epoch)
            if config.REGUL_LOSS_BETA is not None:
                print("Train reg. loss: ",reg_loss_sum)
                wandb.log({"Train reg. loss":reg_loss_sum}, step=epoch)
            #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.h5'))
            #wandb.save(os.path.join(wandb.run.dir, 'model.h5'))
            wandb.watch(self.network)
            if epoch % 20 == 19:
                torch.save(self.network.state_dict(), os.path.join(wandb.run.dir, 'model-{}.h5'.format(epoch)))
                wandb.save(os.path.join(wandb.run.dir, 'model-{}.h5'.format(epoch)))
            
            if epoch % 5 == 4:
                torch.save(self.network.state_dict(), os.path.join(wandb.run.dir, 'model-last.h5'))
                wandb.save(os.path.join(wandb.run.dir, 'model-last.h5'))

            valid_loss=self.evaluate(epoch)

            if config.SCHEDULER_LAMBDA is not None:
                scheduler.step()
            if config.SCHEDULER_PLATEAU:
                scheduler.step(sum_loss) #Valid loss