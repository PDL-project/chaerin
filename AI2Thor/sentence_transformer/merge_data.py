from pathlib import Path
import glob
import pandas as pd

def merge_datasets(name='combined_data.csv',without=['data.csv']):
    cols=['Task', 'Function']
    df = pd.DataFrame(columns=cols)
    csv_files=list(filter(lambda x : 'csv' in x and x not in without and x != name, glob.glob("*")))
    for mini_file in csv_files:
        print(f"Loading {mini_file}...",end='\t')
        mini_name = Path(mini_file).name
        try:
            mini=pd.read_csv(mini_name, sep=",")
        except pd.errors.EmptyDataError as e:
            print(f"{mini_file} is empty")
            continue
        print()
        df = pd.concat([df, mini], axis=0, ignore_index=True)
        df.reset_index()
    df.to_csv(name, index=False)
    return df
    
df=merge_datasets()
print(df.columns)
print(df)
