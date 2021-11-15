from .. import (
    torch, os, plt, clear_output
)

# TODO: Add docstrings!


def visualize_matrices(**matrices):
    n = len(matrices)
    m = len(list(matrices.values())[0])
    figsize = (2 * m + 3, 2 * n + 3)
    fig, axs = plt.subplots(n, m, figsize=figsize)
    if n < 2:
        axs = axs[None, :]
    for i, (k, mat_dict) in enumerate(matrices.items()):
        for j, (k2, M) in enumerate(mat_dict.items()):
            im = axs[i, j].matshow(M, vmin=-1, vmax=1, cmap='nipy_spectral')
            axs[i, j].grid(False)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            axs[i, j].set_title(f'${k2}_{{{k}}}$')
            fig.colorbar(
                im, ax=axs[i, j], fraction=0.024,
                pad=0.01, orientation="vertical", aspect=40
            )


def generate_report(model, epoch, iteration, ldict):
    report = f"m {model:10} e {epoch:4d} i {iteration:6d} "
    for k, v in ldict.items():
        report += f'{k} {v.item():.3E} '
    return report


def plot_results(
    xhs, suptitle, fname=None, cmap='nipy_spectral', vmin=0, vmax=1,
    path='', root='../figures/results', varname='x'
):
    n = len(xhs); h = 2*n + 2
    fig, axs = plt.subplots(n, 1, figsize=(16,h))
    if n == 1:
        axs = [axs]
    for i, (k, xh) in enumerate(xhs.items()):
        img = torch.cat([xh[j] for j in range(8)], axis=1)
        im = axs[i].matshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].grid(False)
        axs[i].set_yticks([])
        axs[i].set_xticks([])
        fig.colorbar(im, ax=axs[i], fraction=0.024, pad=0.01, orientation="vertical", aspect=5)
        _pre = r"\hat" if k != "" else ""
        axs[i].set_title(f"${_pre} {varname}_{{{k}}}$")

    plt.suptitle(suptitle, y=0.96)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(f'{root}/{path}/{fname}.pdf')


def plot_training(
        xhs, epoch, iteration, ldicts, path='', cmap='nipy_spectral', root='../figures/anim',
        save=True, clear=False, add_text = ''
    ):
    n = len(xhs); h = 2*n + 2
    fig, axs = plt.subplots(n, 1, figsize=(16,h))
    if n == 1:
        axs = [axs]
    for i, (model, xh) in enumerate(xhs.items()):
        img = torch.cat([xh[j] for j in range(8)], axis=1)
        im = axs[i].matshow(img, cmap=cmap, vmin=0, vmax=1)
        axs[i].grid(False)
        axs[i].set_yticks([])
        axs[i].set_xticks([])
        fig.colorbar(im, ax=axs[i], fraction=0.024, pad=0.01, orientation="vertical", aspect=5)
        if model in ldicts:
            report = generate_report(model, epoch, iteration, ldicts[model])
        else:
            report = model
        axs[i].set_title(f"{report} {add_text}")

    plt.tight_layout()

    if save:
        try:
            os.makedirs(f'{root}/{path}')
        except:
            pass

        fname = f'{root}/{path}/{epoch}_{iteration}.png'
        plt.savefig(fname)

    if clear:
        clear_output(wait=True)

    plt.pause(1e-6)
