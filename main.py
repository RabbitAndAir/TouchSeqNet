import os

import torch
from args import args
from dataloader_help import create_dataloader
from model.TouchSeqNet import TouchSeqNet
from process import Trainer

def main():

    file_paths = [
        'data/touchalytics/data.csv',
        # 'data/biodent/rawdata.csv',
    ]

    for file_path in file_paths:
        # ✅ 设置随机种子 & 线程数
        torch.set_num_threads(12)
        torch.cuda.manual_seed(3407)

        # ✅ 自动设置 dataset name
        path_parts = file_path.strip().split(os.sep)[1:]
        basename = os.path.splitext(path_parts[-1])[0]
        ds_name = f"{path_parts[0]}_{basename}"
        args.file_path = file_path

        args.save_path = os.path.join('exp/model', ds_name)
        args.save_path_figure = os.path.join('out/figure', ds_name, 'picture')
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path_figure, exist_ok=True)

        print(f'\n🚀 Training on {ds_name}')
        print(f'📦 Model Path: {args.save_path}')
        print(f'🖼️  Figure Path: {args.save_path_figure}')

        train_loader, val_loader = create_dataloader(
            file_path=args.file_path,
            wave_length=args.wave_length,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size
        )

        model = TouchSeqNet(args)
        print('✅ Model initialized.')

        trainer = Trainer(args, model, train_loader, val_loader, verbose=True)

        trainer.pretrain()
        trainer.finetune()

if __name__ == '__main__':
    main()