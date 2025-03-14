import numpy as np
import scipy.io
import scipy.signal
from scipy.signal import find_peaks

# import matplotlib.pyplot as plt


def runupStatistics_CHI(eta, t_sec, nperseg, SS_IG_f_divide):
    # Modified from Dave's version of runupFullStats... and subroutines.

    # Inputs
    #   eta = water level time series; Be sure to subtract tide!
    #   t_sec = timeseries time in seconds
    #   nperseg = window length for spectral analysis
    #   SS_IG_f_divide = frequency cutoff between SS and IG

    # Hardcoded
    #   nbins = 20; number of bins for calculating CDF
    #   MinPeakProminence = np.std(eta)/3; Minimum prominence for find peaks
    #   distance=3; minimum distance between peaks

    #  RUstats.setup = mean(eta); setup
    #         .eta2   = 2# exceedence of eta;
    #         .Tp   = peak period
    #         .Ts   = time/peaks
    #         .Tr   = time/peaks
    #         .Ss   = significant swash = 4*sqrt(var(eta));
    #         .Ssin = significant incident swash
    #         .Ssig = significant infragravity swash
    #         .Sst  = significant total swash

    RUstats = {
        "R2": [],
        "Tr": [],
        "S2": [],
        "Ts": [],
        "setup": [],
        "eta2": [],
        "f": [],
        "S": [],
        "Tp": [],
        "Ss": [],
        "Ssin": [],
        "Ssig": [],
        "Sst": [],
    }

    # sample rate
    dt = np.mean(np.diff(t_sec))

    # Minimum prominence for find peaks
    MinPeakProminence = np.std(eta) / 3

    # Number of bins for cummulative distribution function
    nbins = 20

    ## Setup
    setup = np.mean(eta)
    RUstats["setup"] = setup

    # 2# exceedence value of eta i.e. full time series not peaks
    # Do this using CDF.
    # Define bins.
    c, binCenters = CDF_by_bins(eta, nbins)
    id = np.argmin(np.abs(c - 0.98))
    eta2 = np.interp(0.98, c, binCenters)
    RUstats["eta2"] = eta2

    ## Runup and swash
    # Swash is defined as the range of values between successive zero
    # crossings.
    # find runup peaks
    peaks, _ = find_peaks(eta, prominence=MinPeakProminence, distance=3)
    # plt.plot(eta)
    # plt.plot(peaks, eta[peaks], "x")
    # plt.xlim([0,100])
    # plt.show()

    # 2# exceedence value runup and swash
    R = eta[peaks]
    c, binCenters = CDF_by_bins(R, nbins)
    R2 = np.interp(0.98, c, binCenters)
    # 2% excceedence
    RUstats["R2"] = R2
    RUstats["S2"] = R2 - np.mean(eta)

    # swash period (same as runup period?!)
    RUstats["Ts"] = t_sec[-1] / len(peaks)
    RUstats["Tr"] = t_sec[-1] / len(peaks)

    # Power Spectral Density
    f, S = scipy.signal.welch(
        eta, fs=1 / dt, window="hann", nperseg=nperseg, noverlap=nperseg * (3 / 4)
    )
    RUstats["f"] = f
    RUstats["S"] = S

    # Peak frequency
    fp = f[np.argmax(S)]
    # Peak period.
    Tp = 1 / fp
    RUstats["Tp"] = Tp

    # significant swash = Ss
    Ss = 4 * np.sqrt(np.std(eta))
    RUstats["Ss"] = Ss
    # incident and IG band significant swash = Ssin & Ssig
    df = np.mean(np.diff(f))

    # incident
    varin = sum(S[f > 1 / SS_IG_f_divide]) * df
    # IG
    varig = sum(S[f < 1 / SS_IG_f_divide]) * df
    # Total
    vartot = sum(S) * df

    RUstats["Ssin"] = 4 * np.sqrt(varin)
    RUstats["Ssig"] = 4 * np.sqrt(varig)
    RUstats["Sst"] = 4 * np.sqrt(vartot)

    # Hilary had this at the bottom of runupFullStats.m but didn't use it!?
    # the right way
    inS = 4 * np.sqrt((Ss / 4) ** 2 - varin)
    igS = 4 * np.sqrt((Ss / 4) ** 2 - varig)
    return RUstats


def CDF_by_bins(x, nbins):
    # Do this using CDF.
    # Define bins.
    binWidth = (np.max(x) - np.min(x)) / nbins
    bins = np.arange(np.min(x), np.max(x), binWidth)
    binCenters = bins + 0.5 * binWidth
    c = np.zeros(len(bins))  # cdf
    for ii in np.arange(1, len(bins)).reshape(-1):
        c[ii] = sum(x < bins[ii]) / len(x)
    return c, binCenters


def runup_stats_CHI(Ibp, exc_value=None):
    """
    Created on Nov 30 2022
    Scripts to analyse runup.
    @author: rfonsecadasilva

    Calculate 2% runup from instantaneous water level using zero-downcrossing method.

    Args:
        Ibp (xr data array): Instantaneous (vertical) beach position (in m relative to stil water level).
        exc_value (float): Exception value for runup (abs values greater than exc_value are excluded; in m). Default to None.

    Returns:
        R2 (xr data array): xr data array with R2.
    """
    if exc_value:
        bad_data = (
            xr.apply_ufunc(np.isnan, Ibp.where(lambda x: np.abs(x) > exc_value)).sum()
            / Ibp.size
            * 100
        ).values.item()
        if bad_data != 100:
            print(f"{100-bad_data:.1f} % of runup data is corrupted")
        Ibp = Ibp.where(lambda x: np.abs(x) < exc_value, drop=True)
    time_cross = Ibp.where(
        ((Ibp.shift(t=1) - Ibp.mean(dim="t")) * (Ibp - Ibp.mean(dim="t")) < 0)
        & (Ibp - Ibp.mean(dim="t") > 0),
        drop=True,
    )  # calculate crossing points for zero-downcrossing on swash

    R2 = (
        Ibp.groupby_bins("t", time_cross.t).max(dim="t").quantile(0.98).values
    )  # calculate 2% runup,dim="t_bins"
    print(R2)
    R2 = xr.DataArray(R2, dims=())

    R2.attrs["standard_name"], R2.attrs["long_name"], R2.attrs["units"] = (
        "2% runup",
        "2% runup",
        "m",
    )
    return R2
