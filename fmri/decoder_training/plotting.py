import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_r(cv_rs, save_path):
    df = []
    for dset in cv_rs:
        for i in range(len(cv_rs[dset])):
            for latent_dim in range(cv_rs[dset][0].shape[0]):
                df.append({'Set': dset, 'Latent Dimension': latent_dim + 1,
                           'Correlation (r)': cv_rs[dset][i][latent_dim]})
    df = pd.DataFrame(df)
    ax = sns.boxplot(x='Latent Dimension', y='Correlation (r)', hue='Set', hue_order=['train', 'test'], data=df)
    plt.title('Latent dimension prediction accuracy')
    plt.setp(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig(save_path)
