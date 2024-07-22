from os import path
from pandas import read_csv
from sklearn.metrics import root_mean_squared_error

#-----------------------------------#
#---------- Read the data ----------#
#-----------------------------------#
class MeasurementSamples(object):
    """
    Class to handle the pseudo-code and carrier-phase time series.
    """

    def __init__(self) -> None:
        
        path_data = path.join(
            path.dirname(__file__),
            'measurements', 'simulated_gnss_data.csv'
        )

        self.ts = read_csv(
            filepath_or_buffer=path_data
        )
    
    def get_measurement_at(self, t, to_numpy=True):
        """Returns measurement sample vector at a given time epoch."""

        if to_numpy:
            
            from gnssint.data.constants import dim_meas

            return self.ts.loc[t, :].to_numpy().reshape((dim_meas, 1))
        else:
            return self.ts.loc[t, :]
    
    def get_measurement_comp(self, name_component):
        """Returns the full time series of a single given measurement component"""
        
        return self.ts.loc[:, name_component]

def import_ground_truth(to_numpy=True):
    """
    Import ground truth receiver position.
    """

    path_data = path.join(
        path.dirname(__file__),
        'groundtruth.csv'
    )

    if to_numpy:
        return read_csv(filepath_or_buffer=path_data).to_numpy()
    else:
        return read_csv(filepath_or_buffer=path_data)

#-----------------------------------#
#--------- Evaluate error ----------#
#-----------------------------------#
def rmse_rec_pos(pred_rec_pos):
    """
    Return the RMSE between the receiver position estimate
    and its ground truth.

    Parameters:
    -----------
    pred: array (n_epochs, 2)
    'X' and 'Y' position estimate.

    Returns:
    --------
    array (2,) of RMSE of 'X' and and 'Y' position.
    """

    return root_mean_squared_error(
        y_true=import_ground_truth(to_numpy=True),
        y_pred=pred_rec_pos,
        multioutput='raw_values'
    )
