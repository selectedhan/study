import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import os

from dataset import CustomDataset

class CustomModel(pl.LightningModule):
    def __init__(self, hparams):
        super(CustomModel, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)

        # Model definition
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.Encoder = torch.nn.Sequential(
            torch.nn.Linear(30,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
        )
        self.Decoder = torch.nn.Sequential(
            torch.nn.Linear(128,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,30),
        )

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.data_transforms = transforms.Compose([
                        transforms.Resize((self.hparams.load_size, self.hparams.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.hparams.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.hparams.load_size, self.hparams.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.hparams.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])    

    def forward(self, x_t): # 정의하면 다른 함수에서 self(X)로 사용가능
        # self.init_features()
        # _ = self.model(x_t)
        # return self.features
        X = self.Encoder(x_t)
        X = self.Decoder(x_t)
        return X

    def train_dataloader(self):
        image_datasets = CustomDataset(root=os.path.join(self.hparams.dataset_path,self.hparams.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = CustomDataset(root=os.path.join(self.hparams.dataset_path,self.hparams.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self): # 필수로 def 가 있어야함
        # optimizer1 = Adam(...)
        # optimizer2 = SGD(...)
        # scheduler1 = ReduceLROnPlateau(optimizer1, ...)
        # scheduler2 = LambdaLR(optimizer2, ...)
        # return (
        #     {
        #         "optimizer": optimizer1,
        #         "lr_scheduler": {
        #             "scheduler": scheduler1,
        #             "monitor": "metric_to_track",
        #         },
        #     },
        #     {"optimizer": optimizer2, "lr_scheduler": scheduler2},
        # )
        return Adam(self.parameters(), lr=1e-3)

    def on_train_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        
    def training_step(self, batch, batch_idx): # 학습루프
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # forward 호출 self(입력)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss
    
    def test_step(self, batch, batch_idx): # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.hparams.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))

    def validation_epoch_end(self, outputs):
        # validation 에폭의 마지막에 호출됨
        # outputs은 각 batch마다 validation_step에 return의 배열
        # outputs = [{'loss' : batch_0_loss}, {'loss': batch_1_loss}...]
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss' : avg_loss}
        return {'avg_val_loss' : avg_loss, 'log':tensorboard_logs}
    
    # train, val loop가 동일한 경우 shared_step으로 재사용가능
    def shared_step(self, batch):
        x, y= batch
        ...
        return F.cross_entropy(y_hat, y)
         
    def _shared_eval_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc
        
    # multiple predict dataloaders일때 dataloader_idx인수사용
    # train_step, test_step등에서도 사용가능
    def predict_step(self, batch, batch_idx, dataloader_idx):
        x,y= batch
        x = x.view(x.size(0), -1)
        return self.encoder(x)            

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)