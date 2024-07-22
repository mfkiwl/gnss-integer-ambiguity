from numpy import eye, diag, float64, array
from pandas import DataFrame
from numpy.linalg import norm

#---------- Experimental environment ----------#
# Satellites
n_sats = int(4) # M in the problem
sat1_position = {'X': -1659.55990595, 'Y': 4406.48986884}
sat2_position = {'X': -9997.71250365, 'Y': -3953.34854736}
sat3_position = {'X': -7064.88218366, 'Y': -8153.22810462}
sat4_position = {'X': -6274.79577245, 'Y': -3088.78545914}

sats_position = {
    'sat_1': sat1_position,
    'sat_2': sat2_position,
    'sat_3': sat3_position,
    'sat_4': sat4_position}

sats_position = DataFrame(
    data=list(sats_position.values()),
    index=sats_position.keys(),
    columns=list(sat1_position),
    dtype=None
)

# Measurements
dim_meas = 2*n_sats
dt = 1.0 # seconds
lambda_carr = 19.0 # meters
sigma_code = 100.0 # meters
sigma_carr = 10.0 # meters
n_epochs = int(3600)
# noise measurements
R_noise = eye(N=dim_meas) # dim = code+carrier-phase x n_sats
R_noise *= diag([sigma_code**2,]*n_sats + [sigma_carr**2,]*n_sats)

# State
dim_state = 2+n_sats # dimension of state variable
x0 = array([0.0,]*(dim_state)).reshape((dim_state, 1)) # receiver position + nb of cycles per sat
F_trans = eye(N=dim_state)
sigma_pos = 10.0
sigma_int = 1e-16 # i.e zero-like
# noise covariance
Q_noise = eye(N=dim_state) # dim = receiver position (x,y) + nb of cycles per sat
Q_noise *= diag([sigma_pos,]*2 + [sigma_int,]*n_sats)
