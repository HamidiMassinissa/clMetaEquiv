import matplotlib.pyplot as plt


def plot_cka(cka, save_path=None, title=None):
    """
    Adapted from https://github.com/AntixK/PyTorch-Model-Compare/blob/main/torch_cka/cka.py
    """

    fig, ax = plt.subplots()
    im = ax.imshow(cka['CKA'], origin='lower', cmap='Blues')
    im.norm.autoscale([0.4, 1])
    ax.set_xlabel(f"Layers {cka['model2_name']}", fontsize=15)
    ax.set_ylabel(f"Layers {cka['model1_name']}", fontsize=15)

    if title is not None:
        ax.set_title(f"{title}", fontsize=18)
    else:
        ax.set_title(f"{cka['model1_name']} vs {cka['model2_name']}", fontsize=18)

    plt.colorbar(im)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)


    # plt.show()