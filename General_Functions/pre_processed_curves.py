''' Author: Alex Cohen Dambrós Lopes 

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to pre-process raw data from space satellites
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This is an example code used to obtain light curves from space telescopes and preprocess them automatically

'''

# ============= Imports =============

import os
import queue
import shutil
import numpy as np
import pandas as pd
import lightkurve as lk
from General_Functions import general_data_functions

# ============= Functions =============

def open_datasets(name_telescope, candidates=False):
    
    """
    Description:
        Function used to load data from space telescopes.

    Parameters:
        name_telescope: string
            Name of the telescope you want to load the data.
        candidates : bool
            If True, returns all dataframes with candidate data from the telescope passed as a parameter

    Return:
        pandas.DataFrame
            Dataframe containing candidate target data.
    """
    name_telescope = name_telescope.upper()
    
    # ============= Input validation =============
    if not isinstance(name_telescope, str):
        raise TypeError("name_telescope must be a string.")
    
    if not isinstance(candidates, bool):
        raise TypeError("candidates must be a boolean.")
    
    if name_telescope not in ["K2", "KEPLER", "TESS"]:
        raise ValueError("The telescope name must be one of the following: 'K2', 'Kepler' or 'TESS'.")

    # ============= Open Datasets =============
    
    dataset_telescope = general_data_functions.read_dataset(name_telescope)

   # Defines the information dictionary for each telescope
    telescope_info = {
        "K2": {
            "drop_method": "Radial Velocity",
            "rename_columns": {
                "tic_id": "id_target",
                "pl_orbper": "period",
                "pl_trandur": "duration"
            },
            "select_columns": ['id_target', 'disposition', 'period', 'duration']
        },
        "KEPLER": {
            "drop_method": None,
            "rename_columns": {
                "kepid": "id_target",
                "koi_disposition": "disposition",
                "koi_period": "period",
                "koi_duration": "duration"
            },
            "select_columns": ['id_target', 'disposition', 'period', 'duration', 'koi_time0bk']
        },
        "TESS": {
            "drop_method": None,
            "rename_columns": {
                "tid": "id_target",
                "tfopwg_disp": "disposition",
                "pl_orbper": "period",
                "pl_trandurh": "duration"
            },
            "select_columns": ['id_target', 'disposition', 'period', 'duration']
        }
    }

    # Process the dataset based on the telescope information
    if name_telescope in telescope_info:
        info = telescope_info[name_telescope]
        
        if info['drop_method'] is not None:
            dataset_telescope = dataset_telescope[dataset_telescope['discoverymethod'] != info['drop_method']]
        
        dataset_telescope.rename(columns=info['rename_columns'], inplace=True)
        dataset_telescope = dataset_telescope[info['select_columns']]

    # ============= Save candidate data =============
    
    if candidates:
        candidate_disposition = ['CANDIDATE']
    else:
        # filtering the data by confirmed targets and false positives
        candidate_disposition = ['CONFIRMED', 'FALSE POSITIVE']
    
    
    dataset_telescope = dataset_telescope[dataset_telescope['disposition'].isin(candidate_disposition)]
    
    return dataset_telescope


def saving_preprocessed_data(local_curves, global_curves, local_global_target, candidate = False):
    
    """
    Description:
        This function in Python aims to save the pre-processed data in CSV files. 
        The function takes three parameters: local_curves, global_curves and local_global_target. 
        The first two are the local and global curves and the last one is the label for the pre-processed data. 
        The function creates a directory called "Preprocessed" in the current directory and saves the preprocessed data in CSV files in that directory.

    Parameters:
        local_curves: list of local curves.
        global_curves: list of global curves.
        local_global_target: label for the pre-processed data.
        candidate: bool
        
    Return:
        None.
    """
    
    if candidate:
        local_path = 'Preprocessed_candidate'
        candidate_path = '_candidate'
    else:
        local_path = 'Preprocessed'
        candidate_path = ''
        

    df_local = pd.DataFrame(local_curves)
    df_global = pd.DataFrame(global_curves)

    df_global = df_global.interpolate(axis=1)
    df_local = df_local.interpolate(axis=1)

    df_global['label'] = pd.Series(local_global_target)
    df_local['label'] = pd.Series(local_global_target)

    # ============= Create the directory =============
    
    get_current_path = os.getcwd()

    # Path
    path = os.path.join(get_current_path, local_path)

    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    except OSError as error:
        print("Directory can not be created: ", error)

    df_local.to_csv(path + f'\\preprocessed_local_view{candidate_path}.csv')
    df_global.to_csv(path + f'\\preprocessed_global_view{candidate_path}.csv')


def process_target(name_telescope, row):
    
    """
    Description:
       This function is used to download and pre-process the target data passed as a parameter. Data can be downloaded from TESS, K2 and KEPLER telescopes

    Parameters:
        name_telescope: telescope name.
        row: row containing telescope data.
        
    Return:
        lc_local.flux.value and lc_global.flux.value
        Error = -1
    """

    id_target = row[0]
    period = row[2]
    duration = row[3]
    try:
        t0 = row[4]
    except IndexError:
        t0 = None
        
    try:
        if name_telescope == 'Kepler':
            id_target = 'KIC ' + str(id_target)
        elif name_telescope == 'TESS':
            id_target = 'TIC ' + str(id_target)
        else:
            return - 1
        
        lcs = lk.search_lightcurve(
                id_target, cadence='long').download_all()

        if lcs is not None:

            # This method concatenates all quarters in our LightCurveCollection together, and normalizes them at the same time.
            lc_raw = lcs.stitch()

            # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
            lc_clean = lc_raw.remove_outliers(sigma=20, sigma_upper=4)

            # We have to mask the transit to avoid self-subtraction the genuine planet signal when we flatten the lightcurve. We have to do a hack to find where the time series should be masked.
            if t0 is not None:
                temp_fold = lc_clean.fold(period, epoch_time=t0)
            else:
                temp_fold = lc_clean.fold(period)

            fractional_duration = (duration / 24.0) / period
            phase_mask = np.abs(temp_fold.phase.value) < (
                fractional_duration * 1.5)
            transit_mask = np.in1d(lc_clean.time.value,
                                   temp_fold.time_original.value[phase_mask])

            lc_flat, _ = lc_clean.flatten(
                return_trend=True, mask=transit_mask)

            # Now fold the cleaned, flattened lightcurve:
            if t0 is not None:
                lc_fold = lc_flat.fold(period, epoch_time=t0)
            else:
                lc_fold = lc_flat.fold(period)

            # ============= Defining global curves =============
            lc_global = lc_fold.bin(bins=2001).normalize() - 1
            lc_global = (
                lc_global / np.abs(lc_global.flux.min())) * 2.0 + 1

            phase_mask = (
                lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
            lc_zoom = lc_fold[phase_mask]

            # ============= Defining local curves =============
            lc_local = lc_zoom.bin(bins=201).normalize() - 1
            lc_local = (
                lc_local / np.abs(np.nanmin(lc_local.flux))) * 2.0 + 1

            print(
                f"{id_target} target pre-processing performed, Disposition:{row[1]}")

            return (row[1], lc_local.flux.value, lc_global.flux.value)

        else:
            print("Error downloading target data:", id_target)
            return -1

    except Exception as error:
        print(f"Failed at id: {id_target} | Error: {error}")
        return -1


def process_threads(processinQqueue, answerQueue, finishedTheLines, name_telescope):
    
    """
    Description:
        This function processes queue data in Multithreading.

    Parameters:
        - processinQqueue: Input queue containing the data to be processed.
        - answerQueue: Output queue that receives the processing results.
        - finishTheLines: Flag indicating whether all lines were processed.
        - name_telescope: Accepted name.

    Return:
        None.
    """
    
    while True:
        try:
            row = processinQqueue.get(timeout=0.01)

        except queue.Empty:
            if finishedTheLines.is_set():
                break

            else:
                continue

        if row is None:
            continue

        result = process_target(name_telescope, row)

        if result == -1:
            continue

        answerQueue.put_nowait(result)

    answerQueue.put_nowait("ts")

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= 
Important !!!
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

The 'bin' function present in the current version of the lightkurve 2.3.0 library has an object type error for this reason 
it is necessary to replace the function in the open source library for the code to work correctly! 

This function below is the function present in previous versions of the library. 
It is not possible to go back to old versions of the package in general because there are functions that were not implemented before.

def bin(
        self,
        time_bin_size=None,
        time_bin_start=None,
        n_bins=None,
        aggregate_func=None,
        bins=None,
        binsize=None,
    ):
    
        if binsize is not None and bins is not None:
            raise ValueError("Only one of ``bins`` and ``binsize`` can be specified.")
        elif (binsize is not None or bins is not None) and (
            time_bin_size is not None or n_bins is not None
        ):
            raise ValueError(
                "``bins`` or ``binsize`` conflicts with "
                "``n_bins`` or ``time_bin_size``."
            )
        elif bins is not None:
            if np.array(bins).dtype != np.int:
                raise TypeError("``bins`` must have integer type.")
            elif np.size(bins) != 1:
                raise ValueError("``bins`` must be a single number.")

        if time_bin_start is None:
            time_bin_start = self.time[0]
        if not isinstance(time_bin_start, (Time, TimeDelta)):
            if isinstance(self.time, TimeDelta):
                time_bin_start = TimeDelta(
                    time_bin_start, format=self.time.format, scale=self.time.scale
                )
            else:
                time_bin_start = Time(
                    time_bin_start, format=self.time.format, scale=self.time.scale
                )

        # Backwards compatibility with Lightkurve v1.x
        if time_bin_size is None:
            if bins is not None:
                i = len(self.time) - np.searchsorted(
                    self.time.value, time_bin_start.value - 1e-10
                )
                time_bin_size = (
                    (self.time[-1] - time_bin_start) * i / ((i - 1) * bins)
                ).to(u.day)
            elif binsize is not None:
                i = np.searchsorted(self.time.value, time_bin_start.value - 1e-10)
                time_bin_size = (self.time[i + binsize] - self.time[i]).to(u.day)
            else:
                time_bin_size = 0.5 * u.day
        if not isinstance(time_bin_size, Quantity):
            time_bin_size *= u.day

        # Call AstroPy's aggregate_downsample
        with warnings.catch_warnings():
            # ignore uninteresting empty slice warnings
            warnings.simplefilter("ignore", (RuntimeWarning, AstropyUserWarning))
            ts = aggregate_downsample(
                self,
                time_bin_size=time_bin_size,
                n_bins=n_bins,
                time_bin_start=time_bin_start,
                aggregate_func=aggregate_func,
            )

            # If `flux_err` is populated, assume the errors combine as the root-mean-square
            if np.any(np.isfinite(self.flux_err)):
                rmse_func = (
                    lambda x: np.sqrt(np.nansum(x ** 2)) / len(np.atleast_1d(x))
                    if np.any(np.isfinite(x))
                    else np.nan
                )
                ts_err = aggregate_downsample(
                    self,
                    time_bin_size=time_bin_size,
                    n_bins=n_bins,
                    time_bin_start=time_bin_start,
                    aggregate_func=rmse_func,
                )
                ts["flux_err"] = ts_err["flux_err"]
            # If `flux_err` is unavailable, populate `flux_err` as nanstd(flux)
            else:
                ts_err = aggregate_downsample(
                    self,
                    time_bin_size=time_bin_size,
                    n_bins=n_bins,
                    time_bin_start=time_bin_start,
                    aggregate_func=np.nanstd,
                )
                ts["flux_err"] = ts_err["flux"]

        # Prepare a LightCurve object by ensuring there is a time column
        ts._required_columns = []
        ts.add_column(ts.time_bin_start + ts.time_bin_size / 2.0, name="time")

        # Ensure the required columns appear in the correct order
        for idx, colname in enumerate(self.__class__._required_columns):
            tmpcol = ts[colname]
            ts.remove_column(colname)
            ts.add_column(tmpcol, name=colname, index=idx)

        return self.__class__(ts, meta=self.meta)      
"""