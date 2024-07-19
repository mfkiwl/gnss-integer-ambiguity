from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from gnssint.data import constants
from gnssint.data.utils import MeasurementSamples

# e.g position receiver: [5.517579263180752,-2.4071475774665894]

#-----------------------------#
#------ Setup the filter -----#
#-----------------------------#
filter_naiv = ExtendedKalmanFilter(
    dim_x=constants.dim_state,
    dim_z=constants.dim_meas
)

filter_naiv.x_prior = constants.x0

# error state cov
filter_naiv.Q = constants.Q_noise

# error measurement cov
filter_naiv.R = constants.R_noise
