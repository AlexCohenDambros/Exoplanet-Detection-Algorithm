# Algorithm created to detect exoplanets using the transit method

import itertools
from unittest import result
import lightkurve as lk
import matplotlib.pyplot as plt
from astroquery.mast import Observations
import pprint
import pandas as pd
import numpy as np
import numpy
import warnings
import time
from astropy.stats import sigma_clip
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask, cleaned_array
from scipy import stats

numpy.random.seed(seed=0)  # reproducibility

warnings.filterwarnings('ignore')
pp = pprint.PrettyPrinter(indent=4)

columnTic = pd.read_csv(
    "hlsp_tess-data-alerts_tess_phot_alert-summary-s01+s02+s03+s04_tess_v9_spoc.csv")

target = columnTic["#tic_id"].unique()  # observation target
listTarget = [target]
parameters = itertools.product(*listTarget)


# Function for plotting the results

def plotMethod(results):

    plt.figure()
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(results.periods), numpy.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))

    # ================ new plot ================

    plt.figure()
    plt.plot(results.model_folded_phase,
             results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y,
                color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.48, 0.52)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    # ================ new plot ================
    plt.figure()
    bins = 500
    bin_means, bin_edges, binnumber = stats.binned_statistic(
        results.folded_phase,
        results.folded_y,
        statistic='mean',
        bins=bins)
    bin_stds, _, _ = stats.binned_statistic(
        results.folded_phase,
        results.folded_y,
        statistic='std',
        bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    plt.plot(results.model_folded_phase,
             results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y,
                color='blue', s=10, alpha=0.5, zorder=2)
    plt.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds/2,
        xerr=bin_width/2,
        marker='o',
        markersize=8,
        color='black',
        # capsize=10,
        linestyle='none')
    plt.xlim(0.48, 0.52)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    plt.show()


# Function used to return the results found

def returResults(results):

    print("\n ============================= Results ===================================")
    print('Period of the best-fit signal', format(results.period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:',
          ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.depth, '.5f'))
    print('Best-fit transit duration (in days)',
          format(results.duration, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)

    print("\n - As profundidades de trânsito podem ser usadas para validar um sinal: \n")
    print('Best-fit transit depth (measured at the transit bottom)',
          format(results.depth, '.5f'))
    print('Signal-to-noise ratio of the stacked signal',
          format(results.snr, '.5f'))
    print('Mean depth and uncertainty of even transits (1, 3, ...)',
          results.depth_mean_even)
    print('Mean depth and uncertainty of odd transits (2, 4, ...)',
          results.depth_mean_odd)
    print('Significance (in standard deviations) between odd and even transit depths',
          format(results.odd_even_mismatch, '.2f'))

    print("\n - A significância na diferença entre trânsitos pares e ímpares é baixa, ou seja, ambos são semelhantes. A presença e o significado de trânsitos individuais podem ser estimados em detalhes: \n")
    print('The number of transits', results.transit_count, 'with/without intransit data points:',
          results.distinct_transit_count, '/', results.empty_transit_count)
    print('Number of data points during each unique transit',
          results.per_transit_count)
    print('The mid-transit time for each transit within the time series',
          results.transit_times)
    print('Signal-to-noise ratio per individual transit', results.snr_per_transit)
    print('Signal-to-pink-noise ratio per individual transit',
          results.snr_pink_per_transit)
    

# Main Function TLS - Transit Least Squaresp

def methodTLS(allDatasLighCurve):
    try:
        allDatasLighCurve = allDatasLighCurve.normalize().remove_nans().remove_outliers()
    except:
        allDatasLighCurve = allDatasLighCurve.stitch()

    print("\n", allDatasLighCurve, "\n")

    alltime = allDatasLighCurve.time.value
    allflux = allDatasLighCurve.flux.value

    try:
        model = transitleastsquares(alltime, allflux)
        results = model.power()

        # return datas
        returResults(results)

        # plot graphys
        plotMethod(results)

        # =-=-=-=-=-=-=-=-= Multiple Planets =-=-=-=-=-=-=-=-=
        print("\n =================== Second run =================== \n")

        intransit = transit_mask(
            alltime, results.period, 2*results.duration, results.T0)
        allflux_second_run = allflux[~intransit]
        alltime_second_run = alltime[~intransit]
        alltime_second_run, allflux_second_run = cleaned_array(
            alltime_second_run, allflux_second_run)

        model_second_run = transitleastsquares(
            alltime_second_run, allflux_second_run)
        results_second_run = model_second_run.power()

        # return datas
        returResults(results_second_run)

        # plot graphys
        plotMethod(results_second_run)

    except Exception as e:
        print(e)


start_time = time.time()


# --------------------------------------------------------------------------
# MAIN PROCESS

if __name__ == '__main__':

    for value in parameters:

        try:
            # {value[0]} / TIC 183979262 / TIC 55652896 / Kepler-17 quarter = 10
            # In this list, each line represents a different observation period
            allCurve = lk.search_lightcurve(
                "TIC 55652896", author="SPOC", sector=[3, 4, 5])
            pp.pprint(allCurve)

            # download data from all sectors
            allObservationSectors = allCurve.download_all()
            print("\n", allObservationSectors)

            # method TLS
            dataFrame = methodTLS(allObservationSectors)

        except Exception as error:
            print(error)

        break

    t = time.time() - start_time
    print('\nTime used to process light curves: %f seconds\n' % t)


# Confirm the results or on the website: https://exofop.ipac.caltech.edu/tess/target.php?id=55652896

# tic_id,toi_id,Disposition,RA,Dec,Tmag,Tmag Err,Epoc,Epoc Err,Period,Period Err,Duration,Duration Err
# 55652896,216.01,PC,73.980231,-63.260063,11.504,0.018,1331.28513,0.000867,34.539342,0.001153,5.503218,0.081042
# 55652896,216.02,PC,73.980231,-63.260063,11.504,0.018,1325.335632,0.00306,17.099142,0.001504,2.03145,0.243138