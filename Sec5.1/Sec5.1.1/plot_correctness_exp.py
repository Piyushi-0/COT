import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

experiment_name = 'correctness_exp_full_v3'
df = pd.read_csv(f'logs/{experiment_name}.csv')
df_tabak = pd.read_csv(f'logs/correctness_exp_tabak.csv')

df = pd.DataFrame(df.groupby(by=["n_Z","test z","SEED"]).mean()).reset_index()

def m(Z):
    return 4*(Z-0.5)

def m_dash(Z):
    return -2*(Z-0.5)

def sig(Z):
    return 1

def sig_dash(Z):
    return 8*Z+1

z = np.linspace(0.1,0.9,9)
base = (m(z)-m_dash(z))**2 + (np.sqrt(sig(z))-np.sqrt(sig_dash(z)))**2


font = {'weight' : 'bold' ,
        'size'   : 10}
matplotlib.rc('font', **font)
image_format = 'svg'

fig, axs = plt.subplots(2,4,figsize=(15, 10))

for idx,n_Z in enumerate([100, 200, 400, 800]):
    data = []
    data_tabak = []
    for i in z:
        i = round(i,2)
        tcost = np.array(df.loc[(df['n_Z']==n_Z ) & (np.round(df['test z'],2) == i)]['tcost'])
        tcost_tabak = np.array(df_tabak.loc[(df_tabak['n_Z']==n_Z ) & (np.round(df_tabak['test z'],2) == i)]['tcost'])
        
        data.append(tcost)
        data_tabak.append(tcost_tabak)
    
    for i in range(2):
        axs[i,idx].plot(z,base,'-r')
        axs[i,idx].set_xlim(-0,1)
        axs[i,idx].set_ylim(0,20)
               
    
    bp = axs[0,idx].boxplot(data,positions=z.round(2),widths=.05,showmeans=True,)
    bp_tabak = axs[1,idx].boxplot(data_tabak,positions=z.round(2),widths=.05,showmeans=True,)
    
    for i in range(2):
        axs[i,idx].set_xticklabels(["0.1","","0.3","","0.5","","0.7","","0.9"]) 
    

fig.savefig(f'plots/{experiment_name}.{image_format}', bbox_inches = 'tight',format = image_format ,pad_inches = 0.25,dpi=1200)