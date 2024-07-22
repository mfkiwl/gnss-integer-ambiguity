"""
Online position estimation with Extended Kalman filter from code and carrier-phase
measurements.
The number of cycles of the carrier-phase measurements is estimated as a real number.
"""

from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from gnssint.data import constants
from gnssint.data.utils import MeasurementSamples, rmse_rec_pos
from gnssint.filter.utils import HPos

#-----------------------------#
#------ Setup the filter -----#
#-----------------------------#
filter_naiv = ExtendedKalmanFilter(
    dim_x=constants.dim_state,
    dim_z=constants.dim_meas
)

# initial state
filter_naiv.x = constants.x0

# error state cov
filter_naiv.Q = constants.Q_noise

# error measurement cov
filter_naiv.R = constants.R_noise

filter_naiv.P *= 10.0

#-----------------------------#
#------- Obsveredsignals -----#
#-----------------------------#
code_carrier_samples = MeasurementSamples()
meas_fun = HPos(dim_meas=constants.dim_meas, dim_state=constants.dim_state)

#-----------------------------#
#------ Online filtering -----#
#-----------------------------#
# state estimates along epochs
x_estimate = []

for t in range(int(constants.n_epochs / constants.dt)):

    # data assimilation
    y_t = code_carrier_samples.get_measurement_at(t=t, to_numpy=True)

    # update step: mean of Gaussian x_t|y_t (state posterior)
    filter_naiv.update(z=y_t, HJacobian=meas_fun.h_jacobian, Hx=meas_fun.h_fun)

    # pred step: mean of Gaussian x_t+1|x_t (next state prior)
    filter_naiv.predict()

    # save state
    x_estimate.append(filter_naiv.x_post)

x_estimate = np.array(x_estimate).reshape((constants.n_epochs, constants.dim_state))

#-----------------------------#
#------ Model evaluation -----#
#-----------------------------#
error_rec_pos = rmse_rec_pos(pred_rec_pos=x_estimate[:, :2])
print(f'RMSE = {error_rec_pos}')

#-----------------------------#
#----- Plot pred vs truth ----#
#-----------------------------#
from gnssint.plots import model_eval

fig_resid, ax_resid = model_eval.plot_residual_position(pred=x_estimate[:, :2])

fig_resid.savefig(f'./plots/floatsol_residual_plot.png')