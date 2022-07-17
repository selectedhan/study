import torch
import pytorch_lightning as pl
import argparse

from model import CustomModel

def get_args():
    parser = argparse.ArgumentParser(description='BASELINEWITHPL')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./MVTec')
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_size', type=int, default=256) # 256
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--coreset_sampling_ratio', type=int, default=0.001)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__' :
    args = get_args()
    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = pl.EarlyStopping(
								monitor='val_acc',
                                patience=args.patience,
                                verbose=True,
                                mode='max'
                                )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(...)
    trainer = pl.Trainer.from_argparse_args(
        args
        , default_root_dir=os.path.join(args.project_root_path, args.category)
        , max_epochs=args.num_epochs
        , gpus=1
        # , accelerator='dp' # 2개 gpu에서 분산학습
        # , callbacks = [early_stopping, checkpoint_callback]
        # , progress_bar_refresh_rate = 10
        )
         #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = CustomModel(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)