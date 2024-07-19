from os import path
from pandas import read_csv

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
            return self.ts.loc[t, :].to_numpy()
        else:
            return self.ts.loc[t, :]
    
    def get_measurement_comp(self, name_component):
        """Returns the full time series of a single given measurement component"""
        
        return self.ts.loc[:, name_component]

#-----------------------------------#
#--------- Evaluate error ----------#
#-----------------------------------#
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