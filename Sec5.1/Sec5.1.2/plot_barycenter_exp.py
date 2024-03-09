import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


experiment_name = 'barycenter_exp'
df = pd.read_csv(f'logs/{experiment_name}.csv')
df_tabak = pd.read_csv(f'logs/barycenter_exp_tabak.csv')

df_tabak[''] = 'Tabak et al.'
df[''] = 'Proposed'


# To select result based on the parameter 
# This can be modified according to the experiment hyperparameters

temp_df = df.loc[((df['lr']==2e-2) & ( df['lambda_x'] == 800) & (df['khp_x'] == 1.0) )]
temp_df_tabak = df_tabak.loc[((df_tabak['lr']==1e-3) & ( df_tabak['lambda_x'] == 1e3) )]

joined_df = pd.concat([temp_df_tabak,temp_df])


font = {'weight' : 'bold' ,
        'size'   : 15}
matplotlib.rc('font', **font)
image_format = 'svg'



fig = plt.figure(figsize=(8,6))
box_plot = sns.boxplot(x = joined_df['n_Z'],
            y = joined_df['wd_bc'],
            hue = joined_df[''],
            width = 0.4,
           palette = 'Set1' , )

box_plot.set(xlabel='',ylabel='')

boxplot_fig = box_plot.get_figure()

fig.savefig(f'plots/{experiment_name}.{image_format}', bbox_inches = 'tight',format = image_format ,pad_inches = 0.25,dpi=1200)