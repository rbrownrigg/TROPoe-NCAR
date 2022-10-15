# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015,2022 David D Turner - All Rights Reserved
#
#  This file is part of the "TROPoe" retrieval system.
#
#  TROPoe is free software developed while the author was at NOAA, and is
#  intended to be free software.  It is made available WITHOUT ANY WARRANTY.
#  For more information, contact the author.
#
# ----------------------------------------------------------------------------

# ************************************************************************************
# +
#
#  Abstract:
#      This script is used to create the prior T/q datasets that are needed by the
#    TROPoe retrievals.  It reads in radiosonde data all of the radiosonde
#    data found for the set of months indicated, performs some simple QC on each
#    profile, and:
#    computes the mean state (Xa) and its covariance (Sa).  Th
#    vertical resolution is provided as an input.  If desired, this script will
#    also create the deltaOD model for the IRS (infrared spectrometer) for the 
#    mean profile.  The output is stored as a netCDF file.
#       The sonde files are assumed to be named like ARM radiosondes, and to have the
#    same netCDF file structure.
#
#  Author:
#    Dave Turner
#    Global Systems Laboratory / NOAA
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
from subprocess import Popen, PIPE

import numpy as np
from netCDF4 import Dataset

from Calcs_Conversions import *
from LBLRTM_Functions import *
from Other_functions import *


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

    superadiabatic_maxht = np.array([calc_superadiabatic_maxht(tt[i, :], ht, pp[i, :]) for i in range(nsonde)])

    return {'success': 1, 'secs': ssecs, 'date': datetime.utcnow(),
            'z': ht, 'p': pp, 't': tt, 'w': ww, 'pwv': pwv, 'nsonde': nsonde,
            'superadiabatic_maxht': superadiabatic_maxht}


def calc_superadiabatic_maxht(t, z, p, max_height=1.0):

    # Compute the theta profile
    theta = t2theta(t, np.zeros_like(t), p)

    # Find where theta change with height is negative
    diff = theta[1:] - theta[:-1]
    foo = np.where(diff < 0)[0]

    # Get the max height of the superadiabatic layer
    maxht = -999.
    if len(foo) > 0:
        maxht = np.max(z[1:][foo])

    # Return the value, cross checking with max_height
    if maxht > max_height:
        return max_height
    else:
        return maxht


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
    t = Xa[0:nht].copy()
    w = Xa[nht:2*nht].copy()
    p = inv_hypsometric(z, t + 273.16, surfp)

    if deltaod:
        t += 273.15
        print(Xa)
        rundecker(3, stdatmos, z, p, t, w, mlayers=z, wnum1=300, wnum2=1799.99, tape5=tp5, v10=True, od_only=True)

        command = ' uname -a ; echo $LBL_HOME ; lblrun ' + tp5 + ' ' + lblout+'1'

        process = Popen(command, stdout=PIPE, stderr=PIPE, shell=True, executable='/bin/csh')
        stdout, stderr = process.communicate()

        rundecker(3, stdatmos, z, p, t, w, mlayers=z, wnum1=1800, wnum2=3100, tape5=tp5, v10=True, od_only=True)

        command = ' uname -a ; echo $LBL_HOME ; lblrun ' + tp5 + ' ' + lblout + '2'

        process = Popen(command, stdout=PIPE, stderr=PIPE, shell=True, executable='/bin/csh')
        stdout, stderr = process.communicate()

        # Find the LBLRTM OD files
        files1 = np.array(sorted(glob(os.path.join(lblout+'1', 'OD*'))))
        files2 = np.array(sorted(glob(os.path.join(lblout+'2', 'OD*'))))

        if len(files1) != len(z)-1:
            print("Problem with the LBLRTM calculation")
            return
        if len(files1) != len(files2):
            print("Problem with the ch1 and ch2 LBLRTM calculations")
            return

        # I want to get the appropriate spectral resolution that is
        # consistent with that used in the TROPoe retrieval.  The default is
        # to use the spectral resolution above 8 km

        bar = np.where(z > 8)
        if len(bar[0]) < 0:
            print("This should not happen")
            return
        s1, v1 = lbl_read(files1[bar[0][0]], do_load_data=True)
        s2, v2 = lbl_read(files2[bar[0][0]], do_load_data=True)

        wnum = np.append(v1, v2)
        od = np.zeros((len(files1), len(wnum)))

        # Read in the LBLRTM run
        for i in range(len(files1)):
            s1, v1 = lbl_read(files1[i], do_load_data=True)
            s2, v2 = lbl_read(files2[i], do_load_data=True)

            s = np.append(s1, s2)
            v = np.append(v1, v2)
            od[i, :] = np.interp(wnum, v, s)

        # clean up after ourselves
        command = 'rm -rf ' + lblout + '1 ' + lblout + '2 ' + tp5
        os.system(command)

        # Compute the true monochromatic radiance spectrum
        rad_mono = radxfer(wnum, t, od)

        # Convolve the monochromatic spectrum with the AERI's/ASSIST's instrument function
        b = convolve_to_aeri(wnum, rad_mono)
        awnum = b['wnum']
        arad = b['spec']
        arad = apodizer(arad, 0)

        # Convolve the monochromatic transmission with the AERI's/ASSIST's instrument function
        trans = np.array([np.exp(-1*sum(od[:, i])) for i in range(len(wnum))])
        b = convolve_to_aeri(wnum, trans)
        atrans = apodizer(b['spec'], 0)

        # Now perform some loops over different delta's to find the delta that works
        # best for each IRS channel. I'll do this in an exponential manner. Generate
        # the array of deltas to use
        delta = [0.1, 0.12]
        dmult = 1.05
        while max(delta) < 2:
            delta.append(max(delta) + (delta[-1] - delta[-2]) * dmult)

        # Loop over these deltas, averaging the optical depths at each layer for each,
        # and then do the radxfer
        brad = np.zeros((len(delta), len(awnum)))
        for i in range(len(delta)):
            odb = np.zeros((len(files1), len(awnum)))
            for j in range(len(awnum)):
                foo = np.where((awnum[j]-delta[i] <= wnum) & (wnum <= awnum[j]+delta[i]))
                odb[:, j] = np.sum(od[:, foo[0]], axis=1) / len(foo[0])

            brad[i, :] = radxfer(awnum, t, odb)

        # Find the delta that works best for each IRS spectral element
        crad = np.zeros(len(awnum))
        idx = np.zeros(len(awnum), dtype=int)
        for i in range(len(awnum)):
            sqdiff = (arad[i] - brad[:, i]**2.)
            idx[i] = np.argmin(sqdiff)
            crad[i] = np.squeeze(brad[idx[i], i])

    # Write out the netcdf
    nc = Dataset(outfile, 'w')

    nc.createDimension('height', size=len(z))
    nc.createDimension('height2', size=len(z)*2)

    if deltaod:
        nc.createDimension('wnum', len(awnum))

    var = nc.createVariable('height', 'f4', ('height',))
    var.setncattr('long_name', 'height')
    var.setncattr('units', 'km AGL')
    var[:] = z

    var = nc.createVariable('mean_pressure', 'f4', ('height',))
    var.setncattr('long_name', 'mean_pressure')
    var.setncattr('units', 'mb')
    var[:] = p

    var = nc.createVariable('mean_temperature', 'f4', ('height',))
    var.setncattr('long_name', 'Mean temperature')
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

    if deltaod:
        var = nc.createVariable('wnum', 'f4', ('wnum',))
        var.setncattr('long_name', 'wavenumber')
        var.setncattr('units', 'cm-1')
        var[:] = awnum

        var = nc.createVariable('delta_od', 'f4', ('wnum',))
        var.setncattr('long_name', 'spectral averaging widths')
        var.setncattr('units', 'cm-1')
        var.setncattr('comment', 'The spectral window on each side of the IRS ' +
                      'spectral element that is used for averaging the monochromatic gaseous ' +
                      'optical depths to give the best estimate of the IRS radiance in a fast manner')
        var[:] = np.array(delta)[idx]

        var = nc.createVariable('radiance_true', 'f4', ('wnum',))
        var.setncattr('long_name', 'radiance (truth)')
        var.setncattr('units', 'mW / (m2 sr cm-1)')
        var[:] = arad

        var = nc.createVariable('radiance_fast', 'f4', ('wnum',))
        var.setncattr('long_name', 'radiance computed from fast model using delta_od')
        var.setncattr('units', 'mW / (m2 sr cm-1)')
        var[:] = crad

        var = nc.createVariable('transmittance_true', 'f4', ('wnum',))
        var.setncattr('long_name', 'atmospheric transmittance (truth)')
        var.setncattr('units', 'unitless')
        var[:] = crad

    var = nc.createVariable('superadiabatic_maxht', 'f4', ('sonde_num',))
    var.setncattr('long_name', "Max height of the superadiabatic layer from each sonde")
    var.setncattr('units', 'm agl')
    var[:] = sonde['superadiabatic_maxht']

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
    Sa, Xa = compute_prior(height, '/raid/clamps/clamps/priors/Shreveport/data/', 'SHVsonde', 8, doplot=False, deltaod=True)
