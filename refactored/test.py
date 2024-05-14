import pandas as pd

import Reader
import LogKeysManager
import ParamsExtractor
import DataPreprocessor
import ReduceDim


# =========================
# Testing Reader

file_path = 'DomSpecAn_Refactoring_1/sample_logs_cron_extended.txt'
r = Reader.Reader(file_path)
df = r.read_file()

print(df.head())

# =========================

# =========================
# Testing Params Extractor
log_types = ['cron', 'laurel', 'maillog', 'messages', 'secure', 'user']
file_paths = ['sample_logs/sample_logs_' + logtype + '.txt' for logtype in log_types]

df_list = []

for file_path in file_paths:
    r = Reader.Reader(file_path)
    df = r.read_file()
    df_list.append(df)

#print(df_list[0].head())

df_after = []

for i, df in enumerate(df_list):
    p = ParamsExtractor.ParamsExtractor(df)
    df = p.convert_params(df)
    new_df = p.get_params()
    df_after.append(new_df)


for i in range(len(df_list)):
    #print('=================')
    #print(df_list[i].head())
    #print(df_list[i].columns)
    #print(df_list[i].iloc[0])
    #print('********')
    #print(df_after[i].head())
    #print(df_after[i].columns)
    #print(df_after[i].iloc[0])
    # Concatenate df and df_after
    df_list[i] = pd.concat([df_list[i], df_after[i]], axis=1)
    #print(df_list[i].columns)

# =========================

# =========================
# Testing json log keys file

file_path = 'DomSpecAn_Refactoring_1/log_key.json'
l = LogKeysManager.LogKeysManager(file_path)
log_keys = l.get_log_keys()


for k in log_keys:
    print(k)
    print(log_keys[k])

# We have correctly identified log keys
# =========================

# =========================
# Testing Data Preprcessor

for i, df in enumerate(df_list):
    d = DataPreprocessor.DataPreprocessor(df)
    enc = d.drop_and_one_hot_encode()
    df_list[i] = enc

print(df_list[3].head())

# Works succesfully
# =========================

# =========================
# Testing ReduceDim
r = ReduceDim.ReduceDim(3, df_list[0], [1,0])
pca = r.pca()

# Works succesfully
# =========================

# End of test