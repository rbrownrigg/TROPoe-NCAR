import os
from datetime import datetime, timedelta

import cmocean
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as col
import matplotlib.dates as mdates

#custom cmaps
ptemp_cmap = col.LinearSegmentedColormap.from_list('ptemp_cmap', ['#fef0d9',
                                                                 '#fdcc8a',
                                                                 '#fc8d59',
                                                                 '#d7301f'])

blue_cmap = col.LinearSegmentedColormap.from_list('blue_cmap', ['#f0f9e8',
                                                                '#bae4bc',
                                                                '#7bccc4',
                                                                '#2b8cbe'])
# Get samples of color from both
# the [0 to 1] range of the linspace is how much of the cmap you want to slice
r_col = ptemp_cmap(np.linspace(0.1, 1., 256))
b_col = blue_cmap.reversed()(np.linspace(0.1, 1., 256))
all_colors = np.vstack((b_col,r_col))
temp_cmap = col.LinearSegmentedColormap.from_list('spliced_cmap', all_colors)

# Blue - 60F - Green - 0C - Brown
dwpt_cmap = col.LinearSegmentedColormap.from_list('dwpt_cmap', [(0., '#614623'),
                                                                (.375, '#8f8b1d'),
                                                                (.5, '#16a10e'),
                                                                (.75, '#39b8a9'),
                                                                (1.,'#177fff')]
                                                  )

# Blue 15+, Green 6+, else Brown
wvmr_cmap = col.LinearSegmentedColormap.from_list('', [(0., '#614623'),
                                                       (.25, '#8f8b1d'),
                                                       (.35, '#16a10e'),
                                                       (.75, '#39b8a9'),
                                                       (1.,'#177fff')]
                                                  )

# TB original colormap 
cmaps = {
    'w_hs'     : {'cm': 'seismic',   'label': 'vertical velocity [m/s]'},
    'w_ls'     : {'cm': 'seismic',   'label': 'vertical velocity [m/s]'},
    'wSpd'  : {'cm': 'gist_stern_r',              'label': 'windspeed [m/s]'},
    'wDir'  : {'cm': cmocean.cm.phase,   'label': 'wind direction [deg]'},
    'temp'  : {'cm': temp_cmap, 'label': 'temperature [C]'},
    'theta' : {'cm': ptemp_cmap, 'label': 'Potential temperature [K]'},
    'thetae' : {'cm': ptemp_cmap, 'label': 'Equivalent potential temperature [K]'},
    'wvmr'  : {'cm': wvmr_cmap, 'label': 'Mixing Ratio [g/kg]'},
    'q'     : {'cm': cmocean.cm.haline_r,  'label': 'q [g/kg]'},
    'dewpt' : {'cm': dwpt_cmap,  'label': 'Dewpoint [C]'},
    'rh'    : {'cm': cmocean.cm.haline_r,  'label': 'RH [%]'},
    'std'   : {'cm': cmocean.cm.thermal,  'label': '1-sigma uncertainty'},
    'bSc'   : {'cm': 'magma', 'label': 'Backscatter [log(10) space]'},
    'bSc_TALL' : {'cm': 'magma', 'label': 'Backscatter [log(10) space]'},
    'snr'   : {'cm': cmocean.cm.gray, 'label': 'Intensity (Signal to Noise Ratio)'}
}
# DDT colormap instead
cmaps = {
    'w_hs'     : {'cm': 'seismic',          'label': 'vertical velocity [m/s]'},
    'w_ls'     : {'cm': 'seismic',          'label': 'vertical velocity [m/s]'},
    'wSpd'     : {'cm': 'gist_stern_r',     'label': 'windspeed [m/s]'},
    'wDir'     : {'cm': cmocean.cm.phase,   'label': 'wind direction [deg]'},
    'temp'     : {'cm': cmocean.cm.thermal, 'label': 'temperature [C]'},
    'theta'    : {'cm': cmocean.cm.thermal, 'label': 'Potential temperature [K]'},
    'thetae'   : {'cm': cmocean.cm.thermal, 'label': 'Equivalent potential temperature [K]'},
    'wvmr'     : {'cm': cmocean.cm.thermal, 'label': 'Mixing Ratio [g/kg]'},
    'q'        : {'cm': cmocean.cm.thermal, 'label': 'q [g/kg]'},
    'dewpt'    : {'cm': cmocean.cm.thermal, 'label': 'Dewpoint [C]'},
    'rh'       : {'cm': cmocean.cm.thermal, 'label': 'RH [%]'},
    'std'      : {'cm': cmocean.cm.thermal, 'label': '1-sigma uncertainty'},
    'bSc'      : {'cm': 'magma', 'label': 'Backscatter [log(10) space]'},
    'bSc_TALL' : {'cm': 'magma', 'label': 'Backscatter [log(10) space]'},
    'snr'      : {'cm': cmocean.cm.gray, 'label': 'Intensity (Signal to Noise Ratio)'}
}


class MidPointNormalize(col.Normalize):
    """Defines the midpoint of diverging colormap.
    Usage: Allows one to adjust the colorbar, e.g.
    using contouf to plot data in the range [-3,6] with
    a diverging colormap so that zero values are still white.
    Example usage:
        norm=MidPointNormalize(midpoint=0.0)
        f=plt.contourf(X,Y,dat,norm=norm,cmap=colormap)
     """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        col.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def _fixgap_1d(times, data, gapsize):

    # Find where we have data gaps and create a new time for that gap
    gap_times = np.array([times[i-1]+gapsize for i in range(1, len(times)) if (times[i]-times[i-1]) > gapsize])

    if len(gap_times) == 0:
        return times, data

    # Create an array to append to the good data
    bad = np.full_like(gap_times, np.nan)

    # Append these new things to the actual data and times
    new_data = np.append(data, bad)
    new_times = np.append(times, gap_times)

    # Sort the times and apply this so they're in the proper sequential order
    sort = np.argsort(new_times)

    return new_times[sort], new_data[sort]


def _fixgap_2d(times, data, gapsize):

    times_1d = times[0]

    # Find where we have data gaps and create a new time for that gap
    gap_times = [times_1d[i-1]+gapsize for i in range(1, len(times_1d)) if (times_1d[i]-times_1d[i-1]) > gapsize]
    if len(gap_times) == 0:
        return times, data

    # Tile them so the dimensions are proper in 2d
    gap_times = np.squeeze(np.array([gap_times for i in range(times.shape[0])]))

    # Create an array to append to the good data
    bad = np.full_like(gap_times, np.nan)
    # print(bad.shape)
    # print()
    # Append these new things to the actual data and times
    new_data = np.append(data, bad, axis=1)
    new_times = np.append(times, gap_times, axis=1)

    # Sort the times and apply this so they're in the proper sequential order
    sort = np.argsort(new_times[0])

    return new_times[:, sort], new_data[:, sort]


def fixgap(times, data, gapsize):
    if times.ndim == 1:
        return _fixgap_1d(times, data, gapsize)
    elif times.ndim == 2:
        return _fixgap_2d(times, data, gapsize)
    else:
        raise Exception("Fixing gaps of more than 2 dimensions is not supported...")


def timeheight(time, height, data, field, ax, xlim, ylim, zlim, cmap=None,
                norm=None, min_gap=None, **kwargs):
    '''
    Produces a time height plot of a 2-D field
    :param time: Array of times (1-D or 2-D but must have same dimenstions as height)
    :param height: Array of heights (1-D or 2-D but must have same dimensions as time)
    :param data: Array of the data to plot (2-D)
    :param field: Field being plotted. See dict of options
    :param ax: Axis to plot the data to
    :param datemin: Datetime object
    :param datemax: Datetime object
    :param datamin: Minimum value of data to plot
    :param datamax: Maximum value of data to plot
    :param zmin: Minimum height to plot
    :param zmax: Maximum height to plot
    :return:
    '''

    # Get the colormap and label of the data
    if cmap is None:
        cm, cb_label = cmaps[field]['cm'], cmaps[field]['label']
    else:
        cm, cb_label = cmap, cmaps[field]['label']

    # Convert to grid if not already done
    if time.ndim == 1 and height.ndim == 1:
        time, height = np.meshgrid(time, height)

    # Fill gaps with Nans if needed
    if min_gap is not None:
        time, data = fixgap(time, data, gapsize=min_gap)
        time, height = np.meshgrid(time[0, :], height[:, 0])

    # Create the plot
    if norm is None:
        c = ax.pcolormesh(time, height, data, vmin=zlim[0], vmax=zlim[1], cmap=cm, norm=norm, shading='auto', **kwargs)
    else:
        c = ax.pcolormesh(time, height, data, cmap=cm, norm=norm, shading='auto', **kwargs)

    # Format the colorbar
    c.cmap.set_bad('grey', .5)
    cb = plt.colorbar(c, ax=ax)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # Set the labels
    ax.set_ylabel('Height [m]')
    ax.set_xlabel('Hour [UTC]')
    cb.set_label(cb_label)

    return ax


def timeseries(time, data, sigma, ax, xlim=None, ylim=None, label=None, min_gap=None, **kwargs):

    if min_gap is not None:
        _, data = fixgap(time, data, gapsize=min_gap)
        time, sigma = fixgap(time, sigma, gapsize=min_gap) # If you get errors here, it could be because the sigma isn't the same size as data

    # Create the plot
    ax.plot(time, data, **kwargs, label=label)
    ax.fill_between(time, data-sigma, data+sigma, alpha=.3, **kwargs)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # Set the labels
    ax.set_ylabel('Height [km]')
    ax.set_xlabel('Time [UTC]')

    return ax


def doplot(filename, x_lim=(0, 24), y_lim=(0, 2), temp_lim=(-5, 30), wvmr_lim=(0, 20),
            tuncert_lim=(0, 1.5), wvuncert_lim=(0, 1.5), theta_lim=(290, 320), rh_lim=(0, 100),
           thetae_lim=(310, 330), dewpt_lim=(-15, 25), lwp_thresh=8, min_gap=None, plot_comment=None, prefix=None, outdir=None):

    nc = Dataset(filename)

    # get the times
    time = [datetime.utcfromtimestamp(d) for d in (nc['base_time'][:] + nc['time_offset'][:])]

    # Make the output filenames
    if prefix is not None:
        timeheight_outfile = f"{prefix}.timeheight.{time[0]:%Y%m%d.%H%M%S}.png"
        timeseries_outfile = f"{prefix}.timeseries.{time[0]:%Y%m%d.%H%M%S}.png"
    else:
        timeheight_outfile = f"tropoe.quicklook.timeheight.{time[0]:%Y%m%d.%H%M%S}.png"
        timeseries_outfile = f"tropoe.quicklook.timeseries.{time[0]:%Y%m%d.%H%M%S}.png"

    if outdir is not None:
        timeheight_outfile = os.path.join(outdir, timeheight_outfile)
        timeseries_outfile = os.path.join(outdir, timeseries_outfile)

    # sort the times
    sort = np.argsort(time)
    time = np.array(time)[sort]
    hour = nc['hour'][sort]
    height = nc["height"][:]
    tres = int(nc.VIP_tres)

    # Get data for the time height figures
    temp = nc["temperature"][sort]
    theta = nc["theta"][sort]
    dewpt = nc["dewpt"][sort]
    rh = nc["rh"][sort]
    wvmr = nc["waterVapor"][sort]
    thetae = nc["thetae"][sort]
    sigma_t = nc['sigma_temperature'][sort]
    sigma_wv = nc['sigma_waterVapor'][sort]

    # Get the data for the timeseriess
    try:
        pblh = nc["pblh"][sort]
        sbih = nc["sbih"][sort]
        sigma_pblh = nc["sigma_pblh"][sort]
        sigma_sbih = nc["sigma_sbih"][sort]

        sbLCL = nc["sbLCL"][sort]
        sigma_sbLCL = nc["sigma_sbLCL"][sort]
        mlLCL = nc["mlLCL"][sort]
        sigma_mlLCL = nc["sigma_mlLCL"][sort]

        sbCAPE = nc["sbCAPE"][sort]
        sigma_sbCAPE = nc["sigma_sbCAPE"][sort]
        sbCIN = nc["sbCIN"][sort]
        sigma_sbCIN = nc["sigma_sbCIN"][sort]

        mlCAPE = nc["mlCAPE"][sort]
        sigma_mlCAPE = nc["sigma_mlCAPE"][sort]
        mlCIN = nc["mlCIN"][sort]
        sigma_mlCIN = nc["sigma_mlCIN"][sort]

        rmsr = nc['rmsr'][sort]
    except IndexError:  # For older files so I can cheat and plot them. Should remove before the initial release
        pblh = nc["dindices"][:, 1][sort]
        sigma_pblh = nc["sigma_dindices"][:, 1][sort]

        sbLCL = np.full_like(pblh, np.nan)
        sigma_sbLCL = np.full_like(pblh, np.nan)
        mlLCL = np.full_like(pblh, np.nan)
        sigma_mlLCL = np.full_like(pblh, np.nan)

        sbCAPE = np.full_like(pblh, np.nan)
        sigma_sbCAPE = np.full_like(pblh, np.nan)
        sbCIN = np.full_like(pblh, np.nan)
        sigma_sbCIN = np.full_like(pblh, np.nan)

        mlCAPE = np.full_like(pblh, np.nan)
        sigma_mlCAPE = np.full_like(pblh, np.nan)
        mlCIN = np.full_like(pblh, np.nan)
        sigma_mlCIN = np.full_like(pblh, np.nan)

        rmsr = nc['rmsr'][sort]

    # cbh and quality flag
    cbh = nc["cbh"][sort]
    qcflag = nc["qc_flag"][sort]
    rms = nc["rmsa"][sort]
    hatch = nc['hatchOpen'][sort]
    lwp = nc['lwp'][sort]

    # Don't forget to close the dataset :)
    nc.close()

    # Create the timeheight figure
    fig, axes = plt.subplots(4, 2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    ((t_ax, wv_ax), (tu_ax, wvu_ax), (theta_ax, rh_ax), (thetae_ax, dewpt_ax)) = axes

    norm = MidPointNormalize(midpoint=0.0, vmin=temp_lim[0], vmax=temp_lim[1])
    #t_ax = timeheight(hour, height, temp.T, 'temp', t_ax, x_lim, y_lim, temp_lim, norm=norm)
    t_ax = timeheight(hour, height, temp.T, 'temp', t_ax, x_lim, y_lim, temp_lim, min_gap=min_gap)
    wv_ax = timeheight(hour, height, wvmr.T, 'wvmr', wv_ax, x_lim, y_lim, wvmr_lim, min_gap=min_gap)

    tu_ax = timeheight(hour, height, sigma_t.T, 'std', tu_ax, x_lim, y_lim, tuncert_lim, min_gap=min_gap)
    wvu_ax = timeheight(hour, height, sigma_wv.T, 'std', wvu_ax, x_lim, y_lim, wvuncert_lim, min_gap=min_gap)

    theta_ax = timeheight(hour, height, theta.T, 'theta', theta_ax, x_lim, y_lim, theta_lim, min_gap=min_gap)
    rh_ax = timeheight(hour, height, rh.T, 'rh', rh_ax, x_lim, y_lim, rh_lim, min_gap=min_gap)

    thetae_ax = timeheight(hour, height, thetae.T, 'thetae', thetae_ax, x_lim, y_lim, thetae_lim, min_gap=min_gap)
    dewpt_ax = timeheight(hour, height, dewpt.T, 'dewpt', dewpt_ax, x_lim, y_lim, dewpt_lim, min_gap=min_gap)

    # Add the cbh to all the axes:
    cbh = np.where(lwp > lwp_thresh, cbh, np.nan)
    for ax in axes.flatten():
        ax.scatter(hour, cbh, color='k', marker='.')

    plt.suptitle(f"TROPoe -- {os.path.basename(filename)}")
    plt.figtext(0.98, .01, f"tres = {tres}", ha="center", fontsize=7)
    plt.tight_layout()

    if plot_comment is not None and plot_comment != "":
        plt.subplots_adjust(bottom=.1)
        plt.figtext(0.5, .01, plot_comment, ha="center", fontsize=7,
                    bbox={"facecolor": "yellow", "alpha": 0.5, "pad": 5})

    plt.savefig(timeheight_outfile)
    # plt.show(block=False)

    # Create timeseries figure
    fig, (rms_ax, pbl_ax, lcl_ax, cape_ax, cin_ax) = plt.subplots(5, 1)
    fig.set_figheight(10)
    fig.set_figwidth(7.5)

    rms_ax = timeseries(hour, rmsr, np.zeros_like(rmsr), rms_ax, xlim=x_lim, min_gap=min_gap, color='k',
                        linestyle='-', label="RMSr")
    rms_ax.legend()

    pbl_ax = timeseries(hour, pblh, sigma_pblh, pbl_ax, xlim=x_lim, ylim=y_lim, min_gap=min_gap, color='k',
                        linestyle='-', label="PBLH")
    pbl_ax = timeseries(hour, sbih, sigma_sbih, pbl_ax, xlim=x_lim, ylim=y_lim, min_gap=min_gap, color='r',
                        linestyle='-', label="SBIH")
    pbl_ax.legend()

    lcl_ax = timeseries(hour, sbLCL, sigma_sbLCL, lcl_ax, xlim=x_lim, ylim=y_lim, min_gap=min_gap, color='k',
                        linestyle='-', label="sbLCL")
    lcl_ax = timeseries(hour, mlLCL, sigma_mlLCL, lcl_ax, xlim=x_lim, ylim=y_lim, min_gap=min_gap, color='r',
                        linestyle='-', label="mlLCL")
    lcl_ax.legend()

    cape_ax = timeseries(hour, sbCAPE, sigma_sbCAPE, cape_ax, xlim=x_lim, min_gap=min_gap, color='k',
                        linestyle='-', label="sbCAPE")
    cape_ax = timeseries(hour, mlCAPE, sigma_mlCAPE, cape_ax, xlim=x_lim, min_gap=min_gap, color='r',
                        linestyle='-', label="mlCAPE")
    cape_ax.legend()

    cin_ax = timeseries(hour, sbCIN, sigma_sbCIN, cin_ax, xlim=x_lim, min_gap=min_gap, color='k',
                        linestyle='-', label="sbCIN")
    cin_ax = timeseries(hour, mlCIN, sigma_mlCIN, cin_ax, xlim=x_lim, min_gap=min_gap, color='r',
                        linestyle='-', label="mlCIN")
    cin_ax.legend()
    plt.suptitle(f"TROPoe -- {os.path.basename(filename)}")


    if plot_comment is not None and plot_comment != "":
        plt.subplots_adjust(bottom=.1)
        plt.figtext(0.5, .01, plot_comment, ha="center", fontsize=7,
                    bbox={"facecolor": "yellow", "alpha": 0.5, "pad": 5})

    plt.savefig(timeseries_outfile)
    # plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", action='store', nargs='*')
    parser.add_argument("--x_lim", action='store', default=(0, 24), nargs=2, type=float, help="Hours of day")
    parser.add_argument("--y_lim", action='store', default=(0, 2), nargs=2, type=float, help="Height AGL in km")
    parser.add_argument("--temp_lim", action='store', default=(-10, 20), nargs=2, type=float, help="Temperature (C)")
    parser.add_argument("--wvmr_lim", action='store', default=(0, 20), nargs=2, type=float, help="WVMR (g/kg)")
    parser.add_argument("--tuncert_lim", action='store', default=(0, 3), nargs=2, type=float, help="T uncertainty (C)")
    parser.add_argument("--wvuncert_lim", action='store', default=(0, 3), nargs=2, type=float, help="WVMR uncertainty (g/kg)")
    parser.add_argument("--theta_lim", action='store', default=(290, 320), nargs=2, type=float, help="Theta (K)")
    parser.add_argument("--rh_lim", action='store', default=(0, 100), nargs=2, type=float, help="RH (percent)")
    parser.add_argument("--thetae_lim", action='store', default=(290, 320), nargs=2, type=float, help="Thetae (K)")
    parser.add_argument("--dewpt_lim", action='store', default=(-15, 25), nargs=2, type=float, help="Dewpoint (C)")
    parser.add_argument("--lwp_thresh", action='store', default=8, nargs=1, type=float, help='LWP Threshold for CBH points')
    parser.add_argument("--min_gap", action='store', default=None, nargs=1, type=float, help='Min gap in hours')
    parser.add_argument("--plot_comment", action='store', default=None, type=str, help='Comment to put on the figure')
    args = parser.parse_args()

    for file in args.files:
        doplot(file, args.x_lim, args.y_lim, args.temp_lim, args.wvmr_lim, args.tuncert_lim, args.wvuncert_lim,
               args.theta_lim, args.rh_lim, args.thetae_lim, args.dewpt_lim, args.lwp_thresh, args.min_gap, args.plot_comment)


