''' Author: Alex Cohen Dambrós Lopes 

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to pre-process raw data from space satellites
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This is an example code used to obtain light curves from space telescopes and preprocess them automatically

'''

# ============= Imports =============
import os
import time
import queue
import shutil
import numpy as np
import pandas as pd
import multiprocessing
import lightkurve as lk
from Functions import all_functions
from multiprocessing import Process, Manager, Queue, freeze_support

# ============= Functions =============

def open_datasets(get_candidates=False):
    
    """
    Description:
        Function used to filter all received data from received.

    Parameters:
        get_candidates : bool, optional
            If True, saves the candidate data as a CSV file in a new directory named 'Candidates'.

    Returns:
        pandas.DataFrame
            Dataframe containing candidate target data.
    """

    # ============= Open Datasets =============
    df_tess = all_functions.read_dataset('tess')
    df_kepler = all_functions.read_dataset('kepler')
    df_k2 = all_functions.read_dataset('k2')

    df_kepler.rename(columns={"koi_disposition": "disposition"}, inplace=True)

    # drop rows of planets that were discovered by methods other than transit
    df_k2 = df_k2[df_k2['discoverymethod'] != 'Radial Velocity']

    # Selecting specific columns

    # TESS

    df_tess.rename(columns={"tid": "id_target", 
                            "tfopwg_disp": "disposition", 
                            "pl_orbper": "period",
                            "pl_trandurh": "duration"}, inplace=True)
    df_tess = df_tess[['id_target', 'disposition', 'period', 'duration']]

    # KEPLER
    df_kepler.rename(columns={"kepid": "id_target", 
                              "koi_disposition": "disposition",
                              "koi_period": "period",
                              "koi_duration": "duration"}, inplace=True)
    df_kepler = df_kepler[['id_target', 'disposition', 'period', 'duration', 'koi_time0bk']]

    # K2
    df_k2.rename(columns={"tic_id": "id_target", 
                          "pl_orbper": "period",
                          "pl_trandur": "duration"}, inplace=True)
    
    df_k2 = df_k2[['id_target', 'disposition', 'period', 'duration']]
    
    # ============= save candidate data =============
    if get_candidates:
        df_candidates_tess = df_tess[df_tess['disposition'] == "CANDIDATE"]
        df_candidates_kepler = df_kepler[df_kepler['disposition'] == "CANDIDATE"]
        df_candidates_k2 = df_k2[df_k2['disposition'] == "CANDIDATE"]

        # ============= Create the directory =============
        get_current_path = os.getcwd()

        # Path
        path = os.path.join(get_current_path, 'Candidates')

        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path, exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)

        except OSError as error:
            print("Directory can not be created: ", error)

        df_candidates_tess.to_csv(path + '\\candidate_target_data_TESS.csv')
        df_candidates_kepler.to_csv(path + '\\candidate_target_data_KEPLER.csv')
        df_candidates_k2.to_csv(path + '\\candidate_target_data_K2.csv')

    # filtering the data by confirmed targets and false positives
    candidate_disposition = ['CONFIRMED', 'FALSE POSITIVE']

    df_tess = df_tess[df_tess['disposition'].isin(candidate_disposition)]
    df_kepler = df_kepler[df_kepler['disposition'].isin(candidate_disposition)]
    df_k2 = df_k2[df_k2['disposition'].isin(candidate_disposition)]

    return df_tess, df_kepler, df_k2


def saving_preprocessed_data(local_curves, global_curves, local_global_target):
    
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
        
    Return:
        None.
    """

    df_local = pd.DataFrame(local_curves)
    df_global = pd.DataFrame(global_curves)

    df_global = df_global.interpolate(axis=1)
    df_local = df_local.interpolate(axis=1)

    df_global['label'] = pd.Series(local_global_target)
    df_local['label'] = pd.Series(local_global_target)

    # ============= Create the directory =============
    get_current_path = os.getcwd()

    # Path
    path = os.path.join(get_current_path, 'Preprocessed')

    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    except OSError as error:
        print("Directory can not be created: ", error)

    df_local.to_csv(path + '\\preprocessed_local_view.csv')
    df_global.to_csv(path + '\\preprocessed_global_view.csv')


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
    t0 = row[4] if not pd.isna(row[4]) else None

    try:
        if name_telescope == 'Kepler':
            id_target = 'KIC ' + str(id_target)
            lcs = lk.search_lightcurve(
                id_target, author=name_telescope, cadence='long').download_all()

        elif name_telescope == 'TESS':
            id_target = 'TIC ' + str(id_target)
            lcs = lk.search_lightcurve(
                id_target, mission=name_telescope, cadence='long').download_all()

        else:
            return - 1

        if lcs is not None:

            # This method concatenates all quarters in our LightCurveCollection together, and normalizes them at the same time.
            lc_raw = lcs.stitch()

            # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
            lc_clean = lc_raw.remove_outliers(sigma=3)

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


if __name__ == '__main__':
    
    num_threads = multiprocessing.cpu_count()

    start_time = time.time()

    df_tess, df_kepler, df_k2 = open_datasets(get_candidates= True)
    
    # telescopes_list = {'Kepler': df_kepler, 'TESS': df_tess}

    # TEST
    telescopes_list = {'Kepler': df_kepler.sample(10)}

    # ============= Execution of threads for data pre-processing =============
    local_curves = []
    global_curves = []
    local_global_target = []

    for name_telescope, df_telescope in telescopes_list.items():

        # Manager
        manager = Manager()

        # Flare gun
        finishedTheLines = manager.Event()

        # Processing Queues
        processinQqueue = Queue(df_telescope.shape[0])
        answerQueue = Queue(df_telescope.shape[0] + num_threads)

        threads = []

        for i in range(num_threads):
            threads.append(Process(target=process_threads, args=(
                processinQqueue, answerQueue, finishedTheLines, name_telescope)))
            threads[-1].start()

        for _, row in df_telescope.iterrows():
            processinQqueue.put_nowait(row)

        time.sleep(1)
        finishedTheLines.set()

        threads_finished = 0
        while threads_finished < num_threads:
            try:
                get_result = answerQueue.get(False)
                if get_result == "ts":
                    threads_finished += 1
                    continue

                # Finish processing the data
                (target, data_local, data_global) = get_result
                local_global_target.append(target)
                local_curves.append(data_local)
                global_curves.append(data_global)

            except queue.Empty:
                continue

        for t in threads:
            t.join()

    # marks the end of the runtime
    end_time = time.time()

    # Calculates execution time in seconds
    execution_time = end_time - start_time

    print(f"Runtime: {execution_time:.2f} seconds")

    # Calls the function to save the preprocessed data locally
    saving_preprocessed_data(local_curves, global_curves, local_global_target)


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
