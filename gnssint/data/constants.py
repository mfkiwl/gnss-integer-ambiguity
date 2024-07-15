#---------- Experimental environment ----------#
# Satellites
n_sats = int(4)
sat1_position = {'X': -1659.55990595, 'Y': 4406.48986884}
sat2_position = {'X': -9997.71250365, 'Y': -3953.34854736}
sat3_position = {'X': -7064.88218366, 'Y': -8153.22810462}
sat4_position = {'X': -6274.79577245, 'Y': -3088.78545914}
sats_position = {
    'sat_1': sat1_position,
    'sat_2': sat2_position,
    'sat_3': sat3_position,
    'sat_4': sat4_position}

# Measurements
lambda_carr = 19.0 # meters
sigma_code = 100.0
sigma_carr = 10.0
dt = 1.0 # seconds
