import seaborn as sns
from matplotlib import pyplot as plt

colors = sns.color_palette()


def plot_regret(results, fname='cumreg', folder='data_and_figures'):
    plt.figure()
    plt_num = 0

    for policy in results.keys():
        name = policy
        df = results[name]

        plt.plot(df.index, df.avg, color=colors[plt_num], label=name)
        plt.fill_between(df.index, df.qregret, df.Qregret, alpha=0.2,
                         color=colors[plt_num])

        plt_num += 1
    plt.legend(loc=4)
    plt.ylabel('cumulative regret')
    plt.xlabel('time steps')
    plt.savefig(f'{folder}/{fname}.pdf')  # your local directory
    plt.show()