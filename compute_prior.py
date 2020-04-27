# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015 David D Turner - All Rights Reserved
#
#  This file is part of the "AERIoe" retrieval system.
#
#  AERIoe is free software developed while the author was at NOAA, and is
#  intended to be free software.  It is made available WITHOUT ANY WARRANTY.
#  For more information, contact the author.
#
# ----------------------------------------------------------------------------

# ************************************************************************************
# +
#
#  Abstract:
#      This script is used to create the prior T/q datasets that are needed by the
#    AERIoe and MWRoe retrievals.  It reads in radiosonde data all of the radiosonde
#    data found for the set of months indicated, performs some simple QC on each
#    profile, and:
#    computes the mean state (Xa) and its covariance (Sa).  Th
#    vertical resolution is provided as an input.  If desired, this script will
#    also create the deltaOD model for the AERI for the mean profile.  The output
#    is stored as a netCDF file.
#       The sonde files are assumed to be named like ARM radiosondes, and to have the
#    same netCDF file structure.
#
#  Author:
#    Dave Turner
#    National Severe Storms Laboratory / NOAA
#    Email: dave.turner@noaa.gov
#
#  Ported to Python 3.7 by Tyler Bell (2020)
#
#  Date created:
#    December 2013
#
#  Date last updated:
#    $Date: 2016/11/29 20:01:34 $
# -
# -+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import datetime
from glob import glob

from netCDF4 import Dataset

from Calcs_Conversions import *


def compute_Xa_Sa(sonde, mint, maxt, minq, maxq):
    nZ = len(sonde['z'])
    data = np.zeros((2 * nZ, sonde['nsonde']))

    # Replace any values outside the valid range with the max/min allowable value
    tmpT = np.where(sonde['t'] > maxt, maxt, sonde['t'])
    tmpT = np.where(tmpT < mint, mint, tmpT)
    tmpQ = np.where(sonde['w'] > maxq, maxq, sonde['w'])
    tmpQ = np.where(tmpQ < minq, minq, tmpQ)

    data[0:nZ, :] = tmpT.transpose()
    data[nZ:2 * nZ, :] = tmpQ.transpose()

    Sa = np.cov(data)
    Xa = np.mean(data, axis=1)
    surfp = np.mean(sonde['p'][:, 0])

    return Sa, Xa, surfp


def read_sonde(path, rootname, ht, mm, minpwv, maxpwv):
    print(path)

    print("Reading data in " + path)

    nsonde = 0

    if mm == 0:
        mnth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif mm == -1:
        mnth = [2, 3, 4, 5, 6]
    elif mm == -2:
        mnth = [5, 6, 7, 8, 9]
    elif mm == -3:
        mnth = [8, 9, 10, 11, 12]
    elif mm == -4:
        mnth = [11, 12, 1, 2, 3]
    elif mm == 1:
        mnth = [12, 1, 2]
    elif mm == 12:
        mnth = [11, 12, 1]
    else:
        mnth = [mm - 1, mm, mm + 1]

    # Loop though the files from each month
    for m in mnth:
        filename = f"{path}/*{rootname}*.????{m:02d}??.*.cdf"
        files = glob(filename)

        for fn in files:
            nc = Dataset(fn)

            bt = nc['base_time'][:]
            to = nc['time_offset'][:]
            z = nc['alt'][:]
            p = nc['pres'][:]
            t = nc['tdry'][:]
            u = nc['rh'][:]

            nc.close()

            # Need quick estimate of PWV for QC reasons
            w = rh2w(t, u / 100., p)
            w = np.where(w <= 0., 0.01, w)
            pwv = w2pwv(w, p)

            if pwv < minpwv or pwv > maxpwv:
                continue

            # Do some basic QC
            foo = np.where((u > 0) & (u < 101) & (p > 0) & (p < 1050) & (t > -150) & (t < 50))[0]
            if len(foo) <= 10:
                continue

            z = z[foo]
            p = p[foo]
            t = t[foo]
            u = u[foo]

            # Compute mixing ratio
            w = rh2w(t, u / 100., p)

            # make sure the sonde went high enough
            z = (z - z[0]) / 1000.
            if max(z) >= max(ht):
                ppp = np.zeros((len(ht)))
                ttt = np.zeros((len(ht)))
                www = np.zeros((len(ht)))

                for i in range(len(ht)):
                    if i == 0:
                        minv = 0
                    else:
                        minv = ht[i] - (ht[i] - ht[i - 1]) / 2

                    if i == len(ht) - 1:
                        maxv = ht[i] + (ht[i] - ht[i - 1]) / 2
                    else:
                        maxv = ht[i] + (ht[i + 1] - ht[i]) / 2

                    foo = np.where((minv < z) & (z < maxv))[0]
                    if len(foo) <= 2:
                        ppp[i] = np.interp(ht[i], z, p)
                        ttt[i] = np.interp(ht[i], z, t)
                        www[i] = np.interp(ht[i], z, w)
                    else:
                        ppp[i] = np.exp(np.mean(np.log(p[foo])))
                        ttt[i] = np.mean(t[foo])
                        www[i] = np.mean(w[foo])

                if nsonde == 0:
                    ssecs = bt + to[0]
                    pp = np.transpose(ppp)
                    tt = np.transpose(ttt)
                    ww = np.transpose(www)
                else:
                    ssecs = np.append(ssecs, bt + to[0])
                    tt = np.row_stack((tt, np.transpose(ttt)))
                    pp = np.row_stack((pp, np.transpose(ppp)))
                    ww = np.row_stack((ww, np.transpose(www)))

                nsonde += 1

    if nsonde <= 0:
        print("Error in read_sonde: no radiosonde data files found")
        return {'err': 0}
    #
    # pp = pp.transpose()
    # tt = tt.transpose()
    # ww = ww.transpose()

    pwv = np.array([w2pwv(ww[i, :], pp[i, :]) for i in range(nsonde)])

    return {'success': 1, 'secs': ssecs, 'date': datetime.utcnow(),
            'z': ht, 'p': pp, 't': tt, 'w': ww, 'pwv': pwv, 'nsonde': nsonde}


def compute_prior(z, sonde_path, sonde_rootname, month_idx, outfile=None, deltaod=False, comment="None",
                  minq=.0001, maxq=30, mint=-120, maxt=50, minpwv=0.01, maxpwv=8, tp5='tape5.prior_deltaod',
                  lblout='lblout.prior_deltaod', stdatmos=6, doplot=False):
    if month_idx > 0:
        if outfile is None:
            outfile = f"Xa_Sa_datafile.month_{month_idx:02d}.cdf"
        mnth = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December']
    elif month_idx == 0:
        if outfile is None:
            outfile = 'Xa_Sa_datafile.yearly.cdf'
        mnth = ['Yearly']
    elif month_idx == -1:
        if outfile is None:
            outfile = 'Xa_Sa_datafile.spring.cdf'
        mnth = ['Spring']
    elif month_idx == -2:
        if outfile is None:
            outfile = 'Xa_Sa_datafile.summer.cdf'
        mnth = ['Summer']
    elif month_idx == -3:
        if outfile is None:
            outfile = 'Xa_Sa_datafile.autumn.cdf'
        mnth = ['Autumn']
    elif month_idx == -4:
        if outfile is None:
            outfile = 'Xa_Sa_datafile.winter.cdf'
        mnth = ['Winter']
    else:
        raise Exception("Undefined month_idx value")

    mm = month_idx
    sonde = read_sonde(sonde_path, sonde_rootname, z, mm, minpwv, maxpwv)

    if sonde['success'] == 0:
        print("Error: Unable to successfully read in the radiosonde data -- aborting")
        return

    if doplot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)

        for i in range(sonde['nsonde']):
            ax1.plot(sonde['t'][i], z, 'k', linewidth=1)
            ax2.plot(sonde['w'][i], z, 'k', linewidth=1)

        ax1.set_xlim(mint, maxt)
        ax1.set_title("Sonde T profiles")
        ax1.set_xlabel("Temp [degC]")
        ax1.set_ylabel("Altitude [km AGL]")

        ax2.set_xlim(minq, maxq)
        ax2.set_title("WVMR T profiles")
        ax2.set_xlabel("WVMR [g/kg]")
        ax2.set_ylabel("Altitude [km AGL]")

        plt.tight_layout()
        if month_idx > 0:
            plt.savefig(f'tq_profiles.{month_idx}.png')
        else:
            plt.savefig(f'tq_profiles.{mnth[0]}.png')
        plt.show(block=False)

        plt.figure()
        plt.hist(sonde['pwv'], bins=np.arange(0, 10, .1))
        plt.title("PWV Histogram")
        plt.xlabel("PWV [cm]")
        plt.ylabel('Count')
        plt.xlim(0, 10)
        if month_idx > 0:
            plt.savefig(f'pwv_histogram.{month_idx}.png')
        else:
            plt.savefig(f'pwv_histogram.{mnth[0]}.png')
        plt.show(block=False)

    # Calculate the prior covariance matrix
    Sa, Xa, surfp = compute_Xa_Sa(sonde, mint, maxt, minq, maxq)


    nsonde = sonde['nsonde']
    minpwv = min(sonde['pwv'])
    maxpwv = max(sonde['pwv'])
    avgpwv = np.mean(sonde['pwv'])

    # Build some profiles for the LBLRTM calculation of the prior Xa
    nht = len(z)
    t = Xa[0:nht]
    w = Xa[nht:2*nht]
    p = inv_hypsometric(z, t + 273.16, surfp)

    # TODO - Add in the deltaOD portion of this
    deltaod = False

    # Write out the netcdf
    nc = Dataset(outfile, 'w')

    nc.createDimension('height', size=len(z))
    nc.createDimension('height2', size=len(z)*2)

    # if deltaod:
    #     nc.createDimension('wnum', len(awnum))

    var = nc.createVariable('height', 'f4', ('height',))
    var.setncattr('long_name', 'height')
    var.setncattr('units', 'km AGL')
    var[:] = z

    var = nc.createVariable('mean_pressure', 'f4', ('height',))
    var.setncattr('long_name', 'mean_pressure')
    var.setncattr('units', 'mb')
    var[:] = z

    var = nc.createVariable('mean_temperature', 'f4', ('height',))
    var.setncattr('long_name', 'Mean termperature')
    var.setncattr('units', 'C')
    var[:] = t - 273.15

    var = nc.createVariable('mean_mixingratio', 'f4', ('height',))
    var.setncattr('long_name', 'mean water vapor mixing ratio')
    var.setncattr('units', 'g/kg')
    var[:] = w

    var = nc.createVariable('height2', 'f4', ('height2',))
    var.setncattr('long_name', 'height,height')
    var.setncattr('units', 'km AGL')
    var[:] = np.append(z, z)

    var = nc.createVariable('mean_prior', 'f4', ('height2',))
    var.setncattr('long_name', 'mean prior state (temperature, then WV mixing ratio)')
    var.setncattr('units', 'C, g/kg')
    var[:] = Xa

    var = nc.createVariable('covariance_prior', 'f4', ('height2', 'height2'))
    var.setncattr('long_name', 'covariance of the mean prior')
    var.setncattr('units', 'C, g/kg')
    var[:] = Sa

    nc.setncattr('Date Created', datetime.utcnow().isoformat())
    nc.setncattr('Comment', comment)

    if month_idx < 0:
        value = mnth[0] + ' sonde data were included in this prior dataset'
    else:
        value = 'Sonde data for the month of ' + mnth[month_idx] + \
                ', along with data from the previous and subsequent ' + \
                'months, were included in this prior dataset'

    nc.setncattr('Data_selected', value)
    nc.setncattr('Nprofiles', f"{nsonde} profiles were included in the computation of the prior")
    nc.setncattr('PWV_stats', f"The mean PWV is {avgpwv} cm with a minimum and "
                              f"maximum of {minpwv} cm and {maxpwv} cm, respectively")
    nc.setncattr('QC_limits_T', f"Temperature values must be between {mint} and {maxt} C")
    nc.setncattr('QC_limits_q', f"WV Mixing ratio values must be between {minq} and {maxq} g/kg")


    nc.close()


    return Sa, Xa


if __name__ == "__main__":
    height = np.array([0.0000000e+00, 9.9999998e-03, 2.1000000e-02,
              3.3100002e-02, 4.6410002e-02, 6.1051004e-02,
              7.7156104e-02, 9.4871715e-02, 1.1435889e-01,
              1.3579477e-01, 1.5937425e-01, 1.8531168e-01,
              2.1384284e-01, 2.4522713e-01, 2.7974984e-01,
              3.1772482e-01, 3.5949731e-01, 4.0544704e-01,
              4.5599174e-01, 5.1159096e-01, 5.7275009e-01,
              6.4002514e-01, 7.1402770e-01, 7.9543054e-01,
              8.8497365e-01, 9.8347104e-01, 1.0918182e+00,
              1.2110001e+00, 1.3421001e+00, 1.4863102e+00,
              1.6449413e+00, 1.8194355e+00, 2.0113790e+00,
              2.2225170e+00, 2.4547689e+00, 2.7102461e+00,
              2.9912710e+00, 3.3003983e+00, 3.6404383e+00,
              4.0144825e+00, 4.4259310e+00, 4.8785243e+00,
              5.3763771e+00, 5.9240150e+00, 6.5264168e+00,
              7.1890588e+00, 7.9179649e+00, 8.7197618e+00,
              9.6017380e+00, 1.0571912e+01, 1.1639103e+01,
              1.2813013e+01, 1.4104314e+01, 1.5524745e+01,
              1.7087219e+01])
    Sa, Xa = compute_prior(height, '/Users/tyler.bell/Data/radiosondes/OUN', 'OUNsonde', 2, doplot=False)
