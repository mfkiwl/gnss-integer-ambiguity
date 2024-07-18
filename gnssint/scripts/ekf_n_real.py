from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from gnssint.data import constants
from gnssint.data.utils import MeasurementSamples
from gnssint.filter.utils import HPos

# e.g position receiver: [5.517579263180752,-2.4071475774665894]

rec_pos = np.array([5.517579263180752,-2.4071475774665894])
int_amb = np.array([3, 4, 3, 4]) *1.0

#-----------------------------#
#------ Setup the filter -----#
#-----------------------------#
filter_naiv = ExtendedKalmanFilter(
    dim_x=constants.dim_state,
    dim_z=constants.dim_meas
)

# initial state
filter_naiv.x_prior = constants.x0

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

# print(code_carrier_samples.get_measurement_at(t=0))

# print(
#     meas_fun.h_fun(np.concat((rec_pos, int_amb)))#.reshape((constants.dim_meas, 1))
# )

#-----------------------------#
#------ Online filtering -----#
#-----------------------------#
for t in range(int(constants.n_epochs / constants.dt)):

    y_t = code_carrier_samples.get_measurement_at(t=t, to_numpy=True)
    y_t = y_t.reshape((constants.dim_meas, 1))

    # update step: estimate pdf of x_t|y_t
    filter_naiv.update(z=y_t, HJacobian=meas_fun.h_jacobian, Hx=meas_fun.h_fun)

    # pred step: estimate pdf of x_t|x_t-1
    