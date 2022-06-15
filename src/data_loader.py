import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DirPath

class DataLoader:
    def __init__(self, train=True, x_mean=None, x_std=None, norm=True):
        """
        :param x_mean: If exists, normalize X according to this mean.
        :param x_std: If exists, normalize X according to this std.
        """
        self.x = None
        self.y = None
        self.train = train  # TODO if test required X mean and std
        self.x_mean = x_mean
        self.x_std = x_std
        self.norm = norm

    def prepare_data(self):
        self.load_x_y()
        if self.norm:
            self.norm_data()

    def load_x_y(self):
        data = pd.read_csv(DirPath().input)
        self.x = data.iloc[:, :-1]
        self.y = pd.Series(data.iloc[:, -1], name='y')
        self.x.columns = [col[:col.find('.')] for col in self.x.columns]

    def norm_data(self):
        if self.train:
            self.x_mean = self.x.mean()
            self.x_std = self.x.std()
        self.x = (self.x - self.x_mean) / self.x_std

    def features_distributions(self, log=False):
        with sns.plotting_context('notebook', font_scale=1.5):

            fig_rows = len(self.x.columns)
            fig, ax = plt.subplots(nrows=fig_rows, ncols=1, figsize=(10, 3 * fig_rows))
            for i, feature in enumerate(self.x):
                sns.violinplot(x=self.y, y=self.x[feature], ax=ax[i])
                ax[i].set_xlabel("")
                ax[i].set_ylabel(f"{self.x.columns[i]}")
                if log:
                    ax[i].set_yscale('log')

            fig.suptitle("Numerical Feature Distributions", y=1)
            fig.align_ylabels()
            fig.tight_layout()
            fig.show()
            fig.savefig(rf"{DirPath().output}/Numerical Distributions.pdf", transparent=True)

    def features_correlation(self):
        data = self.x
        for method in ['spearman']:
            corr = data.corr(method=method)
            plt.figure()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", vmin=-1, vmax=1)
            # plt.xticks(rotation=30)
            plt.suptitle(f'Heat Map for {method}', y=1, fontsize='large')
            plt.tight_layout()
            plt.savefig(f'{DirPath().output}/{method}.pdf', transparent=True)
            plt.show()

    def pairs_plot(self):
        data = pd.concat([self.x, self.y], axis=1)
        plt.figure()
        sns.pairplot(data, hue=self.y.name)
        # for i, ax in enumerate(g.axes[-1, :]):
        #    ax.xaxis.set_label_text(lst[i], rotation=20)
        # for i, ax in enumerate(g.axes[:, 0]):
        #    ax.yaxis.set_label_text(lst[i], rotation=0)
        plt.suptitle(f'Pair plot of features', y=2, fontsize='large')
        plt.tight_layout()
        plt.savefig(f'{DirPath().output}/pairs_plot.pdf', transparent=True)
        plt.show()

