from data_process.biodent import create_biodent_sets, Biodent_dataloader
from data_process.ffinger import create_ffinger_sets, Ffinger_dataloader
from args import args
from data_process.touchalytics import create_touchalytics_sets, Touchalytics_dataloader

def create_dataloader(file_path, wave_length, train_batch_size, val_batch_size):
    if file_path == 'data/regulation':
        print('Using regulation dataset')
        train_pairs, val_pairs, max_train_len, max_val_len, data_shape = create_ffinger_sets(file_path=file_path, wave_length=wave_length)
        args.data_shape = data_shape
        train_loader = Ffinger_dataloader(train_pairs, max_len=max_train_len, batch_size=train_batch_size)
        val_loader = Ffinger_dataloader(val_pairs, max_len=max_val_len, batch_size=val_batch_size)

    elif file_path == 'data/biodent/rawdata.csv':
        print('Using biodent dataset')
        train_pairs, val_pairs, max_train_len, max_val_len, data_shape = create_biodent_sets(file_path=file_path,
                                                                                             wave_length=wave_length)
        args.data_shape = data_shape
        train_loader = Biodent_dataloader(train_pairs, max_len=max_train_len, batch_size=train_batch_size)
        val_loader = Biodent_dataloader(val_pairs, max_len=max_val_len, batch_size=val_batch_size)

    elif file_path == 'data/touchalytics/data.csv':
        print('Using touchalytics dataset')
        train_pairs, val_pairs, max_train_len, max_val_len, data_shape = create_touchalytics_sets(file_path=file_path,
                                                                                             wave_length=wave_length)
        args.data_shape = data_shape
        train_loader = Touchalytics_dataloader(train_pairs, max_len=max_train_len, batch_size=train_batch_size)
        val_loader = Touchalytics_dataloader(val_pairs, max_len=max_val_len, batch_size=val_batch_size)

    return train_loader, val_loader