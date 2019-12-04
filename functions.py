import pandas as pd


def processInitialFile(fname, header, filesep):

    df = pd.DataFrame()
    if filesep == 'TXT' and header=='No':
        df = pd.read_csv(fname, sep='\t', header=None)
    elif filesep == 'CSV' and header=='No':
        df = pd.read_csv(fname, header=None)
    elif filesep == 'Excel' and header=='No':
        df = pd.read_excel(fname, header=None)
    elif filesep == 'TXT' and header=='Yes':
        df = pd.read_csv(fname, sep='\t', header=0)
    elif filesep == 'CSV' and header=='Yes':
        df = pd.read_csv(fname, header=0)
    elif filesep == 'Excel' and header=='Yes':
        df = pd.read_excel(fname, header=0)

    return df