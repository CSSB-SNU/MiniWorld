import torch
import matplotlib.pyplot as plt

def plot_expert_frequency_stats(exp_mat, title, save_path=None):
    """
    Plot per-layer expert frequency mean and variance as bar plots.

    Parameters
    ----------
    exp_frequency_params : list[torch.Tensor]
        List of tensors, each of shape (num_experts,).
        Typically length = number of layers (e.g., 28).
    """

    # 레이어별 평균과 분산
    means = exp_mat.mean(dim=1)  # (N_layers,)
    vars_ = exp_mat.var(dim=1)   # (N_layers,)
    layers = range(len(means))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # 평균 bar plot
    axes[0].bar(layers, means, color="tab:blue")
    axes[0].axhline(1/exp_mat.size(1), color="red", linestyle="--", label=f"Uniform (1/{exp_mat.size(1)})")
    axes[0].set_ylabel("Mean")
    axes[0].set_title(f"Per-Layer {title} Mean")
    axes[0].legend()

    # 분산 bar plot
    axes[1].bar(layers, vars_, color="tab:orange")
    axes[1].set_ylabel("Variance")
    axes[1].set_title(f"Per-Layer {title} Variance")
    axes[1].set_xlabel("Layer Index")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else :
        plt.show()
    plt.close()


def plot_expert_frequency_heatmap(exp_mat, title, save_path=None):
    """
    Plot expert frequency distribution across layers as a heatmap.

    Parameters
    ----------
    exp_frequency_params : list[torch.Tensor]
        List of tensors, each of shape (num_experts,).
        Typically length = number of layers (e.g., 28).
    """

    plt.figure(figsize=(8, 6))
    im = plt.imshow(exp_mat, aspect="auto", cmap="viridis")

    # colorbar
    plt.colorbar(im, label=title)

    # 축 라벨
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.title(f"{title} Heatmap")

    # grid, tight layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else :
        plt.show()
    plt.close()

model = torch.load("checkpoints/v0.4.3/MiniWorld_epoch=0070.pt",map_location='cpu',weights_only=False)
params = model['model']
expert_frequency_param_keys = [key for key in params.keys() if 'expert_frequency' in key]
squeeze_param_keys = [key.replace('expert_frequency', 'squeeze_weight') for key in expert_frequency_param_keys]

exp_frequency_params = [params[key] for key in expert_frequency_param_keys]
squeeze_params = [params[key] for key in squeeze_param_keys]

squeeze_param_norm = [squeeze_param.norm(dim=(-1,-2)) for squeeze_param in squeeze_params]
squeeze_param_norm = [squeeze_param/squeeze_param.mean() for squeeze_param in squeeze_param_norm]

exp_mat = torch.stack(exp_frequency_params)  # shape: (28, 8)
squeeze_mat = torch.stack(squeeze_param_norm)  # shape: (28, 8)

plot_expert_frequency_stats(exp_mat, title='Expert Frequency', save_path="expert_frequency_stats.png")
plot_expert_frequency_heatmap(exp_mat, title='Expert Frequency', save_path="expert_frequency_heatmap.png")
plot_expert_frequency_stats(squeeze_mat, title='Ws norm', save_path="squeeze_param_norm_stats.png")
plot_expert_frequency_heatmap(squeeze_mat, title='Ws norm', save_path="squeeze_param_norm_heatmap.png")