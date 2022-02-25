import os
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

k = 5

### chaii
data_dir = 'data/chaii/'

### chaii-trans
data_dir_trans = 'data/chaii-trans/'

for i in tqdm(range(k)):
    ### chaii
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    train_df, test_df = train_test_split(train_df, test_size=100, random_state=i, stratify=train_df['language'])
    train_df, val_df = train_test_split(train_df, test_size=100, random_state=i, stratify=train_df['language'])
    train_ids, test_ids, val_ids = train_df['id'], test_df['id'], val_df['id']

    train_df.to_csv(os.path.join(data_dir, f'train_train_k{i}.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, f'train_test_k{i}.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, f'train_val_k{i}.csv'), index=False)
    with open(os.path.join(data_dir, f'train_ids_k{i}.txt'), 'w') as file:
        file.writelines([f'{id_}\n' for id_ in train_ids.values.tolist()])
    with open(os.path.join(data_dir, f'test_ids_k{i}.txt'), 'w') as file:
        file.writelines([f'{id_}\n' for id_ in test_ids.values.tolist()])
    with open(os.path.join(data_dir, f'val_ids_k{i}.txt'), 'w') as file:
        file.writelines([f'{id_}\n' for id_ in val_ids.values.tolist()])

    ### chaii-trans
    train_translated_df = pd.read_csv(os.path.join(data_dir_trans, 'train_translated.csv'))
    train_transliterated_df = pd.read_csv(os.path.join(data_dir_trans, 'train_transliterated.csv'))

    train_mask = train_translated_df.id.isin(train_ids)
    test_mask = train_translated_df.id.isin(test_ids)
    val_mask = train_translated_df.id.isin(val_ids)
    train_translated_df[train_mask].to_csv(os.path.join(data_dir_trans, f'train_translated_train_k{i}.csv'), index=False)
    train_translated_df[test_mask].to_csv(os.path.join(data_dir_trans, f'train_translated_test_k{i}.csv'), index=False)
    train_translated_df[val_mask].to_csv(os.path.join(data_dir_trans, f'train_translated_val_k{i}.csv'), index=False)


    train_mask = train_transliterated_df.id.isin(train_ids)
    test_mask = train_transliterated_df.id.isin(test_ids)
    val_mask = train_transliterated_df.id.isin(val_ids)
    train_transliterated_df[train_mask].to_csv(os.path.join(data_dir_trans, f'train_transliterated_train_k{i}.csv'), index=False)
    train_transliterated_df[test_mask].to_csv(os.path.join(data_dir_trans, f'train_transliterated_test_k{i}.csv'), index=False)
    train_transliterated_df[test_mask].to_csv(os.path.join(data_dir_trans, f'train_transliterated_val_k{i}.csv'), index=False)