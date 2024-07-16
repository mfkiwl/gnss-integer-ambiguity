from matplotlib import pyplot as plt
from os import path
from gnssint.data.utils import MeasurementSamples
from gnssint.data import constants

data_code_carr = MeasurementSamples()

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax = ax.ravel()

for i in range(1, constants.n_sats + 1):

    # Pseudo-code
    data_code_carr.get_measurement_comp(
        name_component=f'P_sat{i}'
    ).plot(
        ax=ax[0],
        marker='.',
        markersize=2**1,
        linestyle='None',
        color=f'C{i-1}',
        alpha=0.75,
        legend=True
    )

    # Carrier-phase
    data_code_carr.get_measurement_comp(
        name_component=f'Phi_sat{i}'
    ).plot(
        ax=ax[1],
        marker='.',
        markersize=2**1,
        linestyle='None',
        color=f'C{i-1}',
        alpha=0.75
    )

ax[1].set_xlabel('time epoch (sec)')

# save
fig.savefig(
    path.join(
        path.dirname(__file__),
        '..', 'plots', 'raw_measurements.png'
    )
)
