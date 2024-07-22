import matplotlib.pyplot as plt
from gnssint.data.utils import import_ground_truth
from gnssint.data.utils import rmse_rec_pos
from numpy import linspace

def plot_residual_position(pred):
    """
    Plot receiver position estimate vs. ground truth.

    Parameters:
    -----------
    pred: array (n_epochs, 2)
    'X' and 'Y' position estimate.

    Returns
    """
    
    # ground truth position
    x_truth = import_ground_truth(to_numpy=True)

    # RMSE of X and Y positions
    error_rec_pos = rmse_rec_pos(pred_rec_pos=pred[:, :2])

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax = ax.flatten()

    for i, pos in enumerate(['X', 'Y']):

        # scatter of estimate vs. truth
        ax[i].plot(
            pred[:, i], x_truth[:, i],
            linestyle='None',
            marker='.', markersize=2.0,
            color='blue', alpha=0.75
        )

        # identity line
        t = linspace(-50.0, 250.0, num=10**2)
        ax[i].plot(
            t, t,
            linestyle='dashed',
            linewidth=2.0,
            color='orange',
            label='Identity line'
        )

        # decoration
        if pos == 'X':
            fig.legend()

        ax[i].set_xlabel('Estimate')
        ax[i].set_ylabel('Ground truth')
        ax[i].set_title(pos + ' (RMSE = {:.2f} m)'.format(error_rec_pos[i]))

    fig.suptitle('Residual plot (position)')

    return fig, ax
