import os
import pandas as pd

from sklearn.model_selection import train_test_split

### chaii
data_dir = 'data/chaii/'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
train_df, test_df = train_test_split(train_df, test_size=100, random_state=0, stratify=train_df['language'])
train_df, val_df = train_test_split(train_df, test_size=100, random_state=0, stratify=train_df['language'])
train_ids, test_ids, val_ids = train_df['id'], test_df['id'], val_df['id']

train_df.to_csv(os.path.join(data_dir, 'train_train.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'train_test.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'train_val.csv'), index=False)
with open(os.path.join(data_dir, 'train_ids.txt'), 'w') as file:
    file.writelines([f'{id_}\n' for id_ in train_ids.values.tolist()])
with open(os.path.join(data_dir, 'test_ids.txt'), 'w') as file:
    file.writelines([f'{id_}\n' for id_ in test_ids.values.tolist()])
with open(os.path.join(data_dir, 'val_ids.txt'), 'w') as file:
    file.writelines([f'{id_}\n' for id_ in val_ids.values.tolist()])

### chaii-trans
data_dir = 'data/chaii-trans/'

train_translated_df = pd.read_csv(os.path.join(data_dir, 'train_translated.csv'))
train_mask = train_translated_df.id.isin(train_ids)
test_mask = train_translated_df.id.isin(test_ids)
val_mask = train_translated_df.id.isin(val_ids)
train_translated_df[train_mask].to_csv(os.path.join(data_dir, 'train_translated_train.csv'), index=False)
train_translated_df[test_mask].to_csv(os.path.join(data_dir, 'train_translated_test.csv'), index=False)
train_translated_df[val_mask].to_csv(os.path.join(data_dir, 'train_translated_val.csv'), index=False)

train_transliterated_df = pd.read_csv(os.path.join(data_dir, 'train_transliterated.csv'))
train_mask = train_transliterated_df.id.isin(train_ids)
test_mask = train_transliterated_df.id.isin(test_ids)
val_mask = train_transliterated_df.id.isin(val_ids)
train_transliterated_df[train_mask].to_csv(os.path.join(data_dir, 'train_transliterated_train.csv'), index=False)
train_transliterated_df[test_mask].to_csv(os.path.join(data_dir, 'train_transliterated_test.csv'), index=False)
train_transliterated_df[test_mask].to_csv(os.path.join(data_dir, 'train_transliterated_val.csv'), index=False)