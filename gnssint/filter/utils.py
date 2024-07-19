from gnssint.data import constants
from numpy import array, zeros, diag, tile
from numpy.linalg import norm

#---------- Measurement function, h ----------#

def hdist(rec_pos, sat_id):
    """
    Measurement function for pseudo-code:
    l2-norm of satellite#sat_id - receiver (distance).
    
    Parameters:
    -----------
    rec_pos: tuple (x,y)
    Position of the receiver.

    sat_id: int, must be 1, 2, 3 or 4.
    Satellite identifier.

    Returns:
    --------
    Distance in meters between satellite and receiver (float).
    """

    if not len(rec_pos) == 2:
        raise ValueError("rec_pos must be 2-dimensional.")

    if not (isinstance(sat_id, int) and sat_id in [1, 2, 3, 4]):
        raise ValueError("Invalid sattelite identifier, should be integer 1, 2, 3 or 4.")

    sat_pos_ = constants.sats_position.loc[f'sat_{sat_id}'].to_numpy()

    return norm(sat_pos_ - array(rec_pos), ord=None)

def hdist_ambig(rec_pos, sat_ambig, sat_id):
    """Measurement function for carrier-phase."""

    return hdist(rec_pos, sat_id) + constants.lambda_carr * sat_ambig

def grad_hdist(rec_pos, sat_id):
    """
    Gradient of the distance w.r.t the receiver position vector.

    Parameters:
    -----------
    rec_pos: tuple (x,y)
    Position of the receiver.

    sat_id: int, must be 1, 2, 3 or 4.
    Satellite identifier.

    Returns:
    --------
    Gradient vector of the distance (1-d array).
    """
    
    dist = hdist(rec_pos, sat_id)
    der_x = -(constants.sats_position.loc[f'sat_{sat_id}', 'X'] - rec_pos[0])
    der_x /= dist
    der_y = -(constants.sats_position.loc[f'sat_{sat_id}', 'Y'] - rec_pos[1])
    der_y /= dist

    return array([der_x, der_y])

def grad_hdist_ambig(rec_pos, sat_ambig, sat_id):
    """
    Gradient of the distance + lamb_carr*n  w.r.t integer ambiguity 'n'.
    """

    return constants.lambda_carr

class HPos(object):
    """
    Class to handle non-linear measurement function in the extended Kalman filter.
    """

    def __init__(
            self,
            dim_meas=constants.dim_meas,
            dim_state=constants.dim_state
        ) -> None:

        self.dim_meas = dim_meas
        self.dim_sate = dim_state

    def h_fun(self, state):

        u = state.flatten()
        return self._h_fun(rec_pos=u[:2], sat_ambig=u[2:]).reshape((self.dim_meas, 1))

    def _h_fun(self, rec_pos, sat_ambig):
        """
        Returns vector of measurements code and carrier-pahse from state variables.

        Parameters:
        -----------
        rec_pos: tuple (x,y)
        Position of the receiver.

        sat_ambig: tuple of integers
        Number of cycles of the carrier for each satellite.

        Returns:
        --------
        Vector of measurements (2 * #satellites).
        """

        h_pseudcode = [hdist(rec_pos=rec_pos, sat_id=i) for i in range(1, constants.n_sats + 1)]
        h_pseudcode =  array(h_pseudcode)
        # sum
        h_carr = h_pseudcode + constants.lambda_carr*array(sat_ambig)

        # concatenate into vector
        return array((h_pseudcode, h_carr)).flatten()

    def h_jacobian(self, state):
        """
        Returns Jacobian matrix of the measurement function from the state variables.
        """

        # reshape as EKF internally stores 1D array as 2D array with single column
        u = state.flatten()
        return self._h_jacobian(rec_pos=u[:2], sat_ambig=u[2:])

    def _h_jacobian(self, rec_pos, sat_ambig):
         
        # full Jacobian matrix
        H_jacob = zeros(shape=(constants.dim_meas, constants.dim_state))
        M = constants.n_sats

        # Jacobian wrt to receiver position only
        H_jacob_pseudocode = [grad_hdist(rec_pos, sat_id=i) for i in range(1, constants.n_sats + 1)]
        H_jacob_pseudocode = array(H_jacob_pseudocode)

        # block-wise filling
        # first two columns, every rows: Jacobian wrt to position
        H_jacob[:, :2] = tile(H_jacob_pseudocode, (2, 1))

        # last M columns and M rows: Jacobian wrt integer ambiguity
        H_jacob[-M:, -M:] = diag(
            [grad_hdist_ambig(rec_pos, sat_ambig, sat_id=i) for i in range(1, constants.n_sats + 1)]
        )

        # elsewhere, zeros

        return H_jacob