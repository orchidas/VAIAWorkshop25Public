from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft
import spaudiopy as spa
import sofar
from tqdm import tqdm
from loguru import logger

from utils import cart2sph, sph2cart, unpack_coordinates
from spatial_audio.hrtf import HRIRSet


class HRIRReader:

    def __init__(self, sofa_path: Path):
        """
        Read HRIRs from SOFA file.

        Parameters
        ----------
        sofa_path : Path
            Path to the SOFA file containing HRTFs.
        """
        try:
            self.sofa_reader = sofar.read_sofa(str(sofa_path),
                                               verify=True,
                                               verbose=True)
        except Exception as exc:
            raise FileNotFoundError(
                "SOFA file not found in specified location!") from exc

        self.fs = self.sofa_reader.Data_SamplingRate
        self.ir_data = self.sofa_reader.Data_IR
        self.num_dims = self.ir_data.ndim

        logger.info(self.sofa_reader.list_dimensions)
        logger.info(f'Shape of the IRs is {self.ir_data.shape}')

        if self.num_dims > 3:
            self.num_meas, self.num_emitter, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape
        else:
            self.num_meas, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape

        if np.all(self.sofa_reader.ListenerView ==
                  0) or self.sofa_reader.ListenerView.shape[0] == 1:
            self.use_source_view = True
            self._listener_view_type = self.sofa_reader.SourcePosition_Type
            self.listener_view = self.get_source_view(
                self.sofa_reader.SourcePosition_Type)
        else:
            self._listener_view_type = self.sofa_reader.ListenerView_Type
            self.listener_view = self.get_listener_view(
                coord_type=self.listener_view_type)

    @property
    def listener_view_type(self):
        """Return listener view type"""
        return self._listener_view_type

    @listener_view_type.setter
    def listener_view_type(self, coord_sys: str):
        """Set listener view type"""
        self._listener_view_type = coord_sys

    def get_listener_view(self, coord_type: str = "cartesian") -> NDArray:
        """
        Get the listener view array. An array of vectors corresponding to the view direction of the listener.
        This can be in spherical or cartesian coordinates.

        Parameters
        ----------
        coord_type : str, optional
            Required coordinate system (default is "cartesian"). Options: "cartesian" or "spherical".

        Returns
        -------
        NDArray
            Array of listener view vectors in HuRRAh coordinate convention. Dims: [M/I, C].
            M is number of measurements. I is 1. C is 3 (for 3D coordinates).
            Spherical coordinates have angles in degrees.

        Raises
        ------
        ValueError
            If the given coord_type is not one of the supported options or if the SOFA listener view is not in degree, degree, metre units.
        """
        coord_type = coord_type.lower()
        is_listener_cart = self.sofa_reader.ListenerView_Type.lower(
        ) == "cartesian"

        if is_listener_cart:
            list_view_cart = self.sofa_reader.ListenerView
            return list_view_cart
        else:
            # check that we've got angles in degrees
            if self.sofa_reader.ListenerView_Units != "degree, degree, metre":
                raise ValueError(
                    f"Incompatible units for type of ListenerView in SOFA file. "
                    f"Type: {self.sofa_reader.ListenerView_Type}, Units: {self.sofa_reader.ListenerView_Units} "
                    "Should be: degree, degree, metre")
            list_view_sph = self.sofa_reader.ListenerView
            # if radius is set to zero in file, set to 1
            list_view_sph[list_view_sph[:, 2] == 0.0, 2] = 1.0
            az, el, r = unpack_coordinates(list_view_sph, axis=-1)

            # now convert to spherical if needed
            if coord_type == "cartesian":
                list_view_cart = sph2cart(az, el, r, axis=-1, degrees=True)
                return list_view_cart
            else:
                return np.stack((az, el, r), axis=-1)

    def get_source_view(self, coord_type: str = "cartesian") -> NDArray:
        """
        Get the source position array. An array of vectors corresponding to the view direction of the source.
        This can be in spherical or cartesian coordinates.

        Parameters
        ----------
        coord_type : str, optional
            Required coordinate system (default is "cartesian"). Options: "cartesian" or "spherical".

        Returns
        -------
        NDArray
            Array of source view vectors. Dims: [M/I, C].
            M is number of measurements. I is 1. C is 3 (for 3D coordinates).
            Spherical coordinates have angles in degrees.

        Raises
        ------
        ValueError
            If the given coord_type is not one of the supported options or if the SOFA source view is not in degree, degree, metre units.
        """
        coord_type = coord_type.lower()
        is_source_cart = self.sofa_reader.SourcePosition_Type.lower(
        ) == "cartesian"

        if is_source_cart:
            list_view_cart = self.sofa_reader.SourcePosition
        else:
            # check that we've got angles in degrees
            if self.sofa_reader.SourcePosition_Units != "degree, degree, metre":
                raise ValueError(
                    f"Incompatible units for type of SourcePosition in SOFA file. "
                    f"Type: {self.sofa_reader.SourcePosition_Type}, Units: {self.sofa_reader.SourcePosition_Units} "
                    "Should be: degree, degree, metre")
            list_view_sph = self.sofa_reader.SourcePosition
            # if radius is set to zero in file, set to 1
            list_view_sph[list_view_sph[:, 2] == 0.0, 2] = 1.0
            az, el, r = unpack_coordinates(list_view_sph, axis=-1)

        # now convert to spherical if needed
        if coord_type == "cartesian":
            list_view_cart = sph2cart(az, el, r, axis=-1, degrees=True)
            return list_view_cart
        else:
            return np.stack((az, el, r), axis=-1)

    def get_ir_corresponding_to_listener_view(
        self,
        des_listener_view: NDArray,
        axis: int = -1,
        coord_type: str = "spherical",
        degrees: bool = True,
    ) -> NDArray:
        """
        Get IR corresponding to a particular listener view.

        Parameters
        ----------
        des_listener_view : NDArray
            P x 3 array of desired listener views.
        axis : int, optional
            Axis of coordinates (default is -1).
        coord_type : str, optional
            Coordinate system type when specifying listener view (default is "spherical").
        degrees : bool, optional
            Whether the listener view is specified in degrees (default is True).

        Returns
        -------
        NDArray
            P x E x R x N IR corresponding to the particular listener views.
        """
        if axis != -1:
            des_listener_view = des_listener_view.T

        num_views = des_listener_view.shape[0]
        assert num_views < self.num_meas

        # euclidean distance between desired and available views
        dist = np.zeros((self.num_meas, num_views))
        des_ir_matrix = np.zeros((num_views, *self.ir_data.shape[1:]),
                                 dtype=float)

        if coord_type == "spherical":
            az, el, r = unpack_coordinates(des_listener_view.copy(), axis=axis)
            des_listener_view = sph2cart(az, el, r, axis=axis, degrees=degrees)

        # find index of view that minimuses the error from the desied view
        for k in range(num_views):
            dist[:, k] = np.sqrt(
                np.sum((self.listener_view - des_listener_view[k, :])**2,
                       axis=axis))
            closest_idx = np.argmin(dist[:, k])
            des_ir_matrix[k, ...] = self.ir_data[closest_idx, ...]

        return des_ir_matrix


##############################################################
class HRIRWriter:

    def __init__(self,
                 hrir_set: HRIRSet,
                 set_list_view_as_source_pos: bool = False):
        """
        Write HRIRs to SOFA file.
        Args:
            HRIRSet: object of dataclass HRIRSet containing information about the HRIRs, see hrtf.py
            set_list_view_as_source_pos (bool): some plugins require the head orientations as source positions
                                                and not listener view
        """
        self.dims = {
            "R": 2,  # receivers always 2 for HRIRs
            "M": hrir_set.num_rotations,
            "N": hrir_set.ir_len_samps,
            "C": 3,  # coordinate dimension (xyz or aed)
            "I": 1,  # for singleton dimensions
        }
        self.hrir_set = hrir_set
        self.conv = 'SimpleFreeFieldHRIR'
        self.sofa = sofar.Sofa(self.conv)
        self._init_data(set_list_view_as_source_pos)

    def _init_data(self, set_list_view_as_source_pos: bool = False):
        """Initialize the SOFA data fields with correct minimal sizes.
        All arrays are zeros.
        """
        self.dtype = np.float32
        self.sofa.Data_SamplingRate = self.hrir_set.fs
        self.sofa.Data_IR = np.zeros(
            (self.dims["M"], self.dims["R"], self.dims["N"]), self.dtype)
        self.sofa.Data_Delay = np.zeros((self.dims["I"], self.dims["R"]),
                                        self.dtype)
        # listener at origin
        self.sofa.ListenerPosition = np.zeros((self.dims["I"], self.dims["C"]),
                                              self.dtype)
        self.sofa.ListenerView = np.zeros((self.dims["M"], self.dims["C"]),
                                          self.dtype)
        # head facing front
        self.sofa.ListenerUp = np.expand_dims(np.array([0, 0, 1]), axis=0)
        self.sofa.verify()

        # set listener view (SPARTA) or source positions (3DTI)
        if set_list_view_as_source_pos:
            self.set_source_position(self.hrir_set.listener_view,
                                     coordsys=self.hrir_set.listener_view_type)
        else:
            self.set_listener_view(self.hrir_set.listener_view,
                                   coordsys=self.hrir_set.listener_view_type)
        self.set_data(self.hrir_set.hrir_data)

    def set_listener_view(
        self,
        view_mat: NDArray,
        coordsys: str = "cartesian",
    ):
        """Set the listener view vectors. Can be cartesian or spherical. Will be normalized.
        Args:
            view_mat (NDArray): Matrix of view directions. Dims [I/M, C] in SOFA coordinates
            coordsys (str): Coordinate system in which view_mat is provided
        """
        if view_mat.ndim != 2:
            raise ValueError("view_mat must be two-dimensional")
        num_meas, num_coord = view_mat.shape
        if num_meas not in (1, self.dims["M"]):
            raise ValueError(
                "First dimension of view_mat must be either M (num measurements)"
            )
        if num_coord != self.dims["C"]:
            raise ValueError(
                f"Second dimension (coordinates) of view_mat must be size C={self.dims['C']}"
            )

        view_mat = view_mat.astype(self.dtype)
        self.sofa.ListenerView_Type = coordsys.lower()
        is_cartesian = coordsys.lower() == "cartesian"

        if is_cartesian:
            self.sofa.ListenerView = view_mat
            self.sofa.ListenerView_Units = 'metre'
            self.sofa.ListenerView /= np.linalg.norm(self.sofa.ListenerView,
                                                     axis=1,
                                                     keepdims=True)
        else:
            self.sofa.ListenerView = view_mat
            self.sofa.ListenerView_Units = 'degree, degree, metre'

    def set_source_position(self,
                            view_mat: NDArray,
                            coordsys: str = "cartesian"):
        """
        Set source positions
        Args:
            view_mat (NDArray): Matrix of view directions. Dims [I/M, C] in SOFA coordinates
            coordsys (str): Coordinate system in which view_mat is provided
        """
        if view_mat.ndim != 2:
            raise ValueError("view_mat must be two-dimensional")
        num_meas, num_coord = view_mat.shape
        if num_meas not in (1, self.dims["M"]):
            raise ValueError(
                "First dimension of view_mat must be either M (num measurements)"
            )
        if num_coord != self.dims["C"]:
            raise ValueError(
                f"Second dimension (coordinates) of view_mat must be size C={self.dims['C']}"
            )
        view_mat = view_mat.astype(self.dtype)
        is_cartesian = coordsys.lower() == "cartesian"
        self.sofa.SourcePosition_Type = coordsys.lower()

        if is_cartesian:
            self.sofa.SourcePosition = view_mat
            self.sofa.SourcePosition_Units = 'metre'
            self.sofa.SourcePosition /= np.linalg.norm(
                self.sofa.SourcePosition, axis=1, keepdims=True)
        else:
            self.sofa.SourcePosition = view_mat
            self.sofa.SourcePosition_Units = 'degree, degree, metre'

    def set_data(self, hrirs: NDArray, delays: Optional[NDArray] = None):
        """
        Set the HRIR data.
        Args:
            hrirs (NDArray): HRIRs of shape num_measurements x num_receivers x ir_len_samps
            delays (NDArray, optional): ITDs of HRIRs of shape num_measurements x num_receivers
        """
        target_dims = (self.dims["M"], self.dims["R"], self.dims["N"])
        if hrirs.shape != target_dims:
            raise ValueError(
                f"HRIRs array incorrect size (expected: {target_dims}, got: {hrirs.shape})"
            )

        self.sofa.Data_IR = hrirs.astype(self.dtype)

        if delays is not None:
            target_dims_delays = (self.dims["M"], self.dims["R"])
            if delays.shape != target_dims_delays:
                raise ValueError(
                    f"Delays array incorrect size (expected: {target_dims_delays}, got: {delays.shape})"
                )
            self.sofa.Data_Delay = delays.astype(self.dtype)

    def write_to_file(self, filename: str, compression: int = 4):
        """Write the SOFa object to a file.

        Args:
            filename (str): The filename to use.
            compression (int): Amount of data compression used in the underlying HDF5 file.
            The range if 0 (no compression) to 9 (most compression). Defaults to 4.
        """
        self.sofa.verify()
        sofar.write_sofa(filename, self.sofa, compression=compression)
