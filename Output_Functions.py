# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015,2022,2023 by David D Turner, Joshua Gebauer, and Tyler Bell 
#  All Rights Reserved
#
#  This file is part of the "TROPoe" retrieval system.
#
#  TROPoe is free software developed while the authors were at NOAA, and is
#  intended to be free software.  It is made available WITHOUT ANY WARRANTY.
#  For more information, contact the authors.
#
# ----------------------------------------------------------------------------

import os
import numpy as np
import scipy.io
import glob
import sys
from datetime import datetime
from netCDF4 import Dataset

import Other_functions
import Calcs_Conversions
import VIP_Databases_functions

################################################################################
# This file contains the following functions:
# write_example_vip_file()
# add_vip_to_global_atts()
# write_variable()
# write_output()
# create_xret()
################################################################################

################################################################################
# This function will write out an example VIP file, and then stop
################################################################################
def write_example_vip_file(experimental=False, console=False):

    # Grab the full vip from VIP_Databases_functions
    vip = VIP_Databases_functions.full_vip.copy()

    # build a list of vip entries
    # output_lines = [f"{key}: {data['value']} # {data['comment']}" for key, data in vip.items() if data['default'] is True]
    output_lines = []
    for key, data in vip.items():
        # See if the value is a list. If so change to a string to keep things consistent
        if type(data['value']) is list:
            data['value'] = str(data['value'])[1:-1]  # Convert to string and remove the brackets

        if experimental:
            output_lines.append(f"{key} = {data['value']} # {data['comment']}")
        elif not experimental and data['default']:
            output_lines.append(f"{key} = {data['value']} # {data['comment']}")
        else:
            continue

    # Write out the default vip to a file
    with open('default_pyVIP.txt', 'w') as fh:
        for line in output_lines:
            fh.write(line + '\n')

    # If desired, write to console
    if console:
        for l in output_lines:
            print(l)

    sys.exit()

################################################################################
# This function takes all of the entries in the VIP structure, and makes
# them global attributes in the netCDF file that is currently open
################################################################################
def add_vip_to_global_atts(nc, full_vip):
    """
    Function to write out the vip vile into the global attributes
    :param nc: NetCDF4 Dataset object to write the VIP to
    :param vip: Dict containing the vip information
    :return:
    """

    vip = full_vip.copy()  # Copy the VIP so we don't overwrite anything in the next iterations

    for key in sorted(vip.keys()):
        # Grab the data
        data = vip[key]

        # Convert the spectral bands back to a string since we can't save arrays to the attributes
        if key == 'spectral_bands':
            data = ','.join([f"{data[0, i]}-{data[1, i]}" for i in range(len(data[0])) if data[0, i] > 0])

        # Set the netcdf attribute
        nc.setncattr(f"VIP_{key}", data)

    return

################################################################################
# This function writes a variable into a netCDF files for debugging purposes
################################################################################

def write_variable(variable, filename):
    print(f'  DEBUG -- Creating the file {filename:s}')
    dims = np.shape(variable)
    if len(dims) == 0:
        print(f'     The variable is a single variable')
    elif len(dims) == 1:
        print(f'     The variable is a vector')
        print(f'        with a dimension of {dims[0]:d}')
        fid = Dataset(filename, 'w')
        did = fid.createDimension('dim', dims[0])
        var = fid.createVariable('var','f8',('dim',))
        var[:] = variable
        fid.close()
    elif len(dims) == 2:
        print(f'     The variable is a matrix')
        print(f'        with dimensions of {dims[0]:d} and {dims[1]:d}')
        fid = Dataset(filename, 'w')
        did = fid.createDimension('dim1', dims[0])
        did = fid.createDimension('dim2', dims[1])
        var = fid.createVariable('var','f8',('dim1','dim2',))
        var[:] = variable
        fid.close()
    else:
        print('    This is a 3+ dimension variable ??  This code needs to be modified to handle this...')
    return

################################################################################
# This function writes out netCDF files with the output
################################################################################

def write_output(vip, ext_prof, mod_prof, ext_tseries, globatt, xret, prior,
                fsample, derived, dindex, version, exectime, modeflag, nfilename, location,
                cbh_string, shour, verbose):

    success = 0
    # I will replace all temp/WVMR data below the chimney height with this
    # flag
    #nochim = -888.

    # These are the derived indices that I will compute later one. I need to
    # define them here in order to build the netcdf file correctly
    nht = len(xret[0]['z'])
    
    # If fsample is zero, then we will create the netCDF file
    if fsample == 0:
        dt = datetime.utcfromtimestamp(xret[0]['secs'])
        hh = datetime.utcfromtimestamp(xret[0]['secs']).hour
        nn = datetime.utcfromtimestamp(xret[0]['secs']).minute
        ss = datetime.utcfromtimestamp(xret[0]['secs']).second
        hms = hh*10000 + nn*100 + ss

        nfilename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + dt.strftime('%Y%m%d.%H%M%S') + '.nc'
        
        
        if ((os.path.exists(nfilename)) & (vip['output_clobber'] == 0)):
            print('Error: output file exists -- aborting (' + nfilename + ')')
            return success, nfilename
        elif (os.path.exists(nfilename)):
            print('Warning: clobbering existing output file (' +nfilename + ')')

        fid = Dataset(nfilename, 'w')
        tdim = fid.createDimension('time', None)
        hdim = fid.createDimension('height', nht)
        vdim = fid.createDimension('obs_dim', len(xret[0]['dimY']))
        gdim = fid.createDimension('gas_dim', 3)
        ddim = fid.createDimension('dfs_dim', len(xret[0]['dfs']))
        if vip['output_file_keep_small'] == 0:
            adim1 = fid.createDimension('arb_dim1', len(xret[0]['Xn']))
            adim2 = fid.createDimension('arb_dim2', len(xret[0]['Xn']))

        base_time = fid.createVariable('base_time','i4')
        base_time.long_name = 'Epoch time'
        base_time.units = 'seconds since 1970-1-1 00:00:00 0:00'

        time_offset = fid.createVariable('time_offset', 'f8', ('time',))
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 'Seconds'

        time = fid.createVariable('time', 'f8', ('time',))
        time.long_name = 'Time'
        time.units = 'seconds since ' + dt.strftime('%Y-%m-%d') +' 00:00:00 0:00 UTC'
        time.calendar = 'standard'

        hour = fid.createVariable('hour', 'f8', ('time',))
        hour.long_name = 'Time'
        hour.units = 'hours since ' + dt.strftime('%Y-%m-%d') +' 00:00:00 0:00 UTC'

        qc_flag = fid.createVariable('qc_flag', 'i2', ('time',))
        qc_flag.long_name = 'Manual QC flag'
        qc_flag.comment = 'value of 0 implies quality is ok; non-zero values indicate that the sample has suspect quality'
        qc_flag.value_1 = 'Implies hatch was not open for full observing period'
        qc_flag.value_2 = 'Implies retrieval did not converge'
        qc_flag.value_3 = 'Implies retrieval converged but RMS between the observed and computed spectrum is too large'
        qc_flag.RMS_threshold_used_for_QC = str(vip['qc_rms_value']) + ' [unitless]'

        height = fid.createVariable('height', 'f4', ('height',))
        height.long_name = 'Height above ground level'
        height.units = 'km'

        temperature = fid.createVariable('temperature', 'f4', ('time','height',))
        temperature.long_name = 'Temperature'
        temperature.units = 'C'

        waterVapor = fid.createVariable('waterVapor', 'f4', ('time', 'height',))
        waterVapor.long_name = 'Water vapor mixing ratio'
        waterVapor.units = 'g/kg'

        lwp = fid.createVariable('lwp', 'f4', ('time',))
        lwp.long_name = 'Liquid water path'
        lwp.units = 'g/m2'

        lReff = fid.createVariable('lReff', 'f4', ('time',))
        lReff.long_name = 'Liquid water effective radius'
        lReff.units = 'microns'

        iTau = fid.createVariable('iTau', 'f4', ('time',))
        iTau.long_name = 'Ice cloud optical depth (geometric limit)'

        iReff = fid.createVariable('iReff', 'f4', ('time',))
        iReff.long_name = 'Ice effective radius'
        iReff.units = 'microns'

        co2 = fid.createVariable('co2', 'f4', ('time','gas_dim',))
        co2.long_name = 'Carbon dioxide concentration'
        co2.units = 'ppm'

        ch4 = fid.createVariable('ch4', 'f4', ('time', 'gas_dim',))
        ch4.long_name = 'Methane concentration'
        ch4.units = 'ppm'

        n2o = fid.createVariable('n2o', 'f4', ('time', 'gas_dim',))
        n2o.long_name = 'Nitrous oxide concentration'
        n2o.units = 'ppm'

        sigmaT = fid.createVariable('sigma_temperature', 'f4', ('time','height',))
        sigmaT.long_name = '1-sigma uncertainty in temperature'
        sigmaT.units = 'C'

        sigmaWV = fid.createVariable('sigma_waterVapor', 'f4', ('time','height',))
        sigmaWV.long_name = '1-sigma uncertainty in water vapor mixing vapor'
        sigmaWV.units = 'g/kg'

        sigma_lwp = fid.createVariable('sigma_lwp', 'f4', ('time',))
        sigma_lwp.long_name = '1-sigma uncertainty in liquid water path'
        sigma_lwp.units = 'g/m2'

        sigma_lReff = fid.createVariable('sigma_lReff', 'f4', ('time',))
        sigma_lReff.long_name = '1-sigma uncertainty in liquid water effective radius'
        sigma_lReff.units = 'microns'

        sigma_iTau = fid.createVariable('sigma_iTau', 'f4', ('time',))
        sigma_iTau.long_name = '1-sigma uncertainty in ice cloud optical depth (geometric limit)'

        sigma_iReff = fid.createVariable('sigma_iReff', 'f4', ('time',))
        sigma_iReff.long_name = '1-sigma uncertainty in ice effective radius'
        sigma_iReff.units = 'microns'

        sigma_co2 = fid.createVariable('sigma_co2', 'f4', ('time','gas_dim',))
        sigma_co2.long_name = '1-sigma uncertainty in carbon dioxide concentration'
        sigma_co2.units = 'ppm'

        sigma_ch4 = fid.createVariable('sigma_ch4', 'f4', ('time','gas_dim',))
        sigma_ch4.long_name = '1-sigma uncertainty in methane concentration'
        sigma_ch4.units = 'ppm'

        sigma_n2o = fid.createVariable('sigma_n2o', 'f4', ('time','gas_dim',))
        sigma_n2o.long_name = '1-sigma uncertaintiy in nitrous oxide concentration'
        sigma_n2o.units = 'ppm'

        converged_flag = fid.createVariable('converged_flag', 'i2', ('time',))
        converged_flag.long_name = 'Convergence flag'
        converged_flag.value_0 = '0 indicates no convergence'
        converged_flag.value_1 = '1 indicates convergence in Rodgers sense (i.e., di2m << dimY)'
        converged_flag.value_2 = '2 indicates convergence (best rms after rms increased drastically'
        converged_flag.value_3 = '3 indicates convergence (best rms after max_iter)'
        converged_flag.value_9 = '9 indicates found NaN in Xnp1'

        gamma = fid.createVariable('gamma', 'f4', ('time',))
        gamma.long_name = 'Gamma parameter'

        n_iter = fid.createVariable('n_iter', 'i2', ('time',))
        n_iter.long_name = 'Number of iterations performed'

        rmsr = fid.createVariable('rmsr', 'f4', ('time',))
        rmsr.long_name = 'Root mean square error between IRS and MWR obs in the observation vector and the forward calculation'
        rmsr.comment1 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        rmsr.comment2 = 'Only IRS radiance observations in the observation vector are used'

        rmsa = fid.createVariable('rmsa', 'f4', ('time',))
        rmsa.long_name = 'Root mean square error between observation vector and the forward calculation'
        rmsa.comment1 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        rmsa.comment2 = 'Entire observation vector used in this calculation'

        rmsp = fid.createVariable('rmsp', 'f4', ('time',))
        rmsp.long_name = 'Root mean square error between prior T/q profile and the retrieved T/q profile'
        rmsp.comment1 = 'Computed as sqrt( mean[ ((Xa - Xn) / sigma_Xa)^2 ] )'

        chi2 = fid.createVariable('chi2', 'f4', ('time',))
        chi2.long_name = 'Chi-square statistic of Y vs. F(Xn)'
        chi2.comment = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'

        convergence_criteria = fid.createVariable('convergence_criteria', 'f4', ('time',))
        convergence_criteria.long_name = 'Convergence criteria di^2'

        dfs = fid.createVariable('dfs', 'f4', ('time','dfs_dim',))
        dfs.long_name = 'Degrees of freedom of signal'
        dfs.comment = 'total DFS, then DFS for each of temperature, waterVapor, LWP, L_Reff, I_tau, I_Reff, carbonDioxide, methane, nitrousOxide'

        sic = fid.createVariable('sic', 'f4', ('time',))
        sic.long_name = 'Shannon information content'

        vres_temp = fid.createVariable('vres_temperature', 'f4', ('time','height',))
        vres_temp.long_name = 'Vertical resolution of the temperature profile'
        vres_temp.units = 'km'
        vres_wv = fid.createVariable('vres_waterVapor', 'f4', ('time','height',))
        vres_wv.long_name = 'Vertical resolution of the water vapor profile'
        vres_wv.units = 'km'

        cdfs_temp = fid.createVariable('cdfs_temperature', 'f4', ('time','height',))
        cdfs_temp.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for temperature'
        cdfs_wv = fid.createVariable('cdfs_waterVapor', 'f4', ('time','height',))
        cdfs_wv.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for water vapor'

        hatchOpen = fid.createVariable('hatchOpen', 'i2', ('time',))
        hatchOpen.long_name = 'Flag indicating if the IRSs hatch was open'
        hatchOpen.comment = '1 - hatch open, 0 - hatch closed, other values indicate hatch is either not working or indeterminant'

        cbh = fid.createVariable('cbh', 'f4', ('time',))
        cbh.long_name = 'Cloud base height above ground level'
        cbh.units = 'km'

        cbh_flag = fid.createVariable('cbh_flag', 'i2', ('time',))
        cbh_flag.long_name = 'Flag indicating the source of the cbh'
        cbh_flag.comment1 = 'Value 0 implies Clear Sky radiance'
        cbh_flag.comment2 = 'Value 1 implies Inner Window radiance'
        cbh_flag.comment3 = 'Value 2 implies Outer Window radiance'
        cbh_flag.comment4 = 'Value 3 implies Default CBH radiance'

        pressure = fid.createVariable('pressure', 'f4', ('time','height',))
        pressure.long_name = 'Derived pressure'
        pressure.units = 'mb'
        pressure.comment = 'derived from IRS surface pressure observations and the hyposmetric calculation using the thermodynamic profiles'

        theta = fid.createVariable('theta', 'f4', ('time','height',))
        theta.long_name = 'Potential temperature'
        theta.units = 'K'
        theta.comment = 'This field is derived from the retrieved fields'

        thetae = fid.createVariable('thetae', 'f4', ('time','height',))
        thetae.long_name = 'Equivalent potential temperature'
        thetae.units = 'K'
        thetae.comment = 'This field is derived from the retrieved fields'

        rh = fid.createVariable('rh', 'f4', ('time','height',))
        rh.long_name = 'Relative humidity'
        rh.units = '%'
        rh.comment = 'This field is derived from the retrieved field'

        dewpt = fid.createVariable('dewpt', 'f4', ('time','height',))
        dewpt.long_name = 'Dew point temperature'
        dewpt.units = 'C'
        dewpt.comment = 'This field is derived from the retrieved fields'

        pwv = fid.createVariable('pwv', 'f4', ('time',))
        pwv.long_name = 'Precipitable water vapor'
        pwv.units = dindex['units'][0]
        pwv.comment1 = 'This field is derived from the retrieved fields'
        pwv.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        pblh = fid.createVariable('pblh', 'f4', ('time',))
        pblh.long_name = 'Planetary boundary layer height'
        pblh.units = dindex['units'][1]
        pblh.comment1 = 'This field is derived from the retrieved fields'
        pblh.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbih = fid.createVariable('sbih', 'f4', ('time',))
        sbih.long_name = 'Surface-based inversion height'
        sbih.units = dindex['units'][2]
        sbih.comment1 = 'This field is derived from the retrieved fields'
        sbih.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbim = fid.createVariable('sbim', 'f4', ('time',))
        sbim.long_name = 'Surface-based inversion magnitude'
        sbim.units = dindex['units'][3]
        sbim.comment1 = 'This field is derived from the retrieved fields'
        sbim.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sblcl = fid.createVariable('sbLCL', 'f4', ('time',))
        sblcl.long_name = 'Lifted condesation level for a surface-based parcel'
        sblcl.units = dindex['units'][4]
        sblcl.comment1 = 'This field is derived from the retrieved fields'
        sblcl.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbcape = fid.createVariable('sbCAPE', 'f4', ('time',))
        sbcape.long_name = 'Convective available potential energy for a surface-based parcel'
        sbcape.units = dindex['units'][5]
        sbcape.comment1 = 'This field is derived from the retrieved fields'
        sbcape.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbcin = fid.createVariable('sbCIN', 'f4', ('time',))
        sbcin.long_name = 'Convective inhibition for a surface-based parcel'
        sbcin.units = dindex['units'][6]
        sbcin.comment1 = 'This field is derived from the retrieved fields'
        sbcin.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mllcl = fid.createVariable('mlLCL', 'f4', ('time',))
        mllcl.long_name = 'Lifted condesation level for a mixed-layer parcel'
        mllcl.units = dindex['units'][7]
        mllcl.comment1 = 'This field is derived from the retrieved fields'
        mllcl.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mlcape = fid.createVariable('mlCAPE', 'f4', ('time',))
        mlcape.long_name = 'Convective available potential energy for a mixed-layer parcel'
        mlcape.units = dindex['units'][8]
        mlcape.comment1 = 'This field is derived from the retrieved fields'
        mlcape.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mlcin = fid.createVariable('mlCIN', 'f4', ('time',))
        mlcin.long_name = 'Convective inhibition for a mixed-layer parcel'
        mlcin.units = dindex['units'][9]
        mlcin.comment1 = 'This field is derived from the retrieved fields'
        mlcin.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'

        sigma_pwv = fid.createVariable('sigma_pwv', 'f4', ('time',))
        sigma_pwv.long_name = '1-sigma uncertainties in precipitable water vapor'
        sigma_pwv.units = dindex['units'][0]
        sigma_pwv.comment1 = 'This field is derived from the retrieved fields'
        sigma_pwv.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_pwv.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'

        sigma_pblh = fid.createVariable('sigma_pblh', 'f4', ('time',))
        sigma_pblh.long_name = '1-sigma uncertainties in the PBL height'
        sigma_pblh.units = dindex['units'][1]
        sigma_pblh.comment1 = 'This field is derived from the retrieved fields'
        sigma_pblh.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_pblh.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbih = fid.createVariable('sigma_sbih', 'f4', ('time',))
        sigma_sbih.long_name = '1-sigma uncertainties in the surface-based inversion height'
        sigma_sbih.units = dindex['units'][2]
        sigma_sbih.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbih.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbih.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbim = fid.createVariable('sigma_sbim', 'f4', ('time',))
        sigma_sbim.long_name = '1-sigma uncertainties in the surface-based inversion magnitude'
        sigma_sbim.units = dindex['units'][3]
        sigma_sbim.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbim.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbim.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sblcl = fid.createVariable('sigma_sbLCL', 'f4', ('time',))
        sigma_sblcl.long_name = '1-sigma uncertainties in the LCL for a surface-based parcel'
        sigma_sblcl.units = dindex['units'][4]
        sigma_sblcl.comment1 = 'This field is derived from the retrieved fields'
        sigma_sblcl.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sblcl.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbcape = fid.createVariable('sigma_sbCAPE', 'f4', ('time',))
        sigma_sbcape.long_name = '1-sigma uncertainties in surface-based CAPE'
        sigma_sbcape.units = dindex['units'][5]
        sigma_sbcape.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbcape.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbcape.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbcin = fid.createVariable('sigma_sbCIN', 'f4', ('time',))
        sigma_sbcin.long_name = '1-sigma uncertainties in surface-based CIN'
        sigma_sbcin.units = dindex['units'][6]
        sigma_sbcin.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbcin.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbcin.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mllcl = fid.createVariable('sigma_mlLCL', 'f4', ('time',))
        sigma_mllcl.long_name = '1-sigma uncertainties in the LCL for a mixed-layer parcel'
        sigma_mllcl.units = dindex['units'][7]
        sigma_mllcl.comment1 = 'This field is derived from the retrieved fields'
        sigma_mllcl.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mllcl.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mlcape = fid.createVariable('sigma_mlCAPE', 'f4', ('time',))
        sigma_mlcape.long_name = '1-sigma uncertainties in mixed-layer CAPE'
        sigma_mlcape.units = dindex['units'][8]
        sigma_mlcape.comment1 = 'This field is derived from the retrieved fields'
        sigma_mlcape.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mlcape.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mlcin = fid.createVariable('sigma_mlCIN', 'f4', ('time',))
        sigma_mlcin.long_name = '1-sigma uncertainties in mixed-layer CIN'
        sigma_mlcin.units = dindex['units'][9]
        sigma_mlcin.comment1 = 'This field is derived from the retrieved fields'
        sigma_mlcin.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mlcin.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        obs_flag = fid.createVariable('obs_flag', 'i2', ('obs_dim',))
        obs_flag.long_name = 'Flag indicating type of observation for each vector element'

        # This will make sure that I capture all of the units right in
        # the metadata, but "blotting them out" as I add the comments

        marker = np.copy(xret[0]['flagY'])
        foo = np.where(xret[0]['flagY'] == 1)[0]
        if len(foo) > 0:
            obs_flag.value_01 = 'Radiance in wavenumber -- i.e., cm^(-1)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 2)[0]
        if len(foo) > 0:
            obs_flag.value_02 = 'Brightness temperature in K from a zenith-microwave radiometer'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 3)[0]
        if len(foo) > 0:
            obs_flag.value_03 = 'Temperature in ' + ext_prof['tunit'] + ' from ' + ext_prof['ttype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 4)[0]
        if len(foo) > 0:
            obs_flag.value_04 = 'Water vapor in ' + ext_prof['qunit'] + ' from ' + ext_prof['qtype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 5)[0]
        if len(foo) > 0:
            obs_flag.value_05 = 'Surface met temeprature in ' + ext_tseries['tunit'] + ' from ' + ext_tseries['ttype']
            obs_flag.value_05_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
            if ext_tseries['sfc_temp_rep_error'] > 0:
                obs_flag.value_05_comment2 = 'Adding ' + str(ext_tseries['sfc_temp_rep_error']) + ' C to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 6)[0]
        if len(foo) > 0:
            obs_flag.value_06 = 'Surface met water vapor in ' + ext_tseries['qunit'] + ' from ' + ext_tseries['qtype']
            obs_flag.value_06_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
            if np.abs(ext_tseries['sfc_wv_mult_error']-1) > 0.01:
                obs_flag.value_06_comment2 = 'Multiplying by ' + str(ext_tseries['sfc_wv_mult_error'])
            if ext_tseries['sfc_wv_rep_error'] > 0:
                obs_flag.value_06_comment2 = 'Adding ' + str(ext_tseries['sfc_wv_rep_error']) + ' g/kg to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 7)[0]
        if len(foo) > 0:
            obs_flag.value_07 = 'Temperature in ' + mod_prof['tunit'] + ' from ' + mod_prof['ttype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 8)[0]
        if len(foo) > 0:
            obs_flag.value_08 = 'Water vapor in ' + mod_prof['qunit'] + ' from '  + mod_prof['qtype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 9)[0]
        if len(foo) > 0:
            obs_flag.value_09 = 'CO2 in-situ obs in ' + ext_tseries['co2unit'] + ' from ' + ext_tseries['co2type']
            if ext_tseries['co2_sfc_rep_error'] > 0:
                obs_flag.value_09_comment1 = 'Adding ' + ext_tseries['co2_sfc_rep_error'] + ' ppm to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 10)[0]
        if len(foo) > 0:
            obs_flag.value_10 = 'Brightness temperature in K from a scanning microwave radiometer'
            obs_flag.value_10_comment1 = 'Dimension is coded to be (frequency[GHz]*100)+(elevation_angle[deg]/1000)'
            marker[foo] = -1

        # If there were any values for marker that were not treated above,
        # then the code must assume that I've screwed up and should abort.

        foo = np.where(marker >= 0)[0]
        if len(foo) > 0:
            print('Error in write_output: there seems to be a unit that is not handled here properly')
            return success, nfilename

        obs_dimension = fid.createVariable('obs_dimension', 'f8', ('obs_dim',))
        obs_dimension.long_name = 'Dimension of the observation vector'
        obs_dimension.comment1 = 'mixed units -- see obs_flag field above'

        obs_vector = fid.createVariable('obs_vector', 'f4', ('time','obs_dim',))
        obs_vector.long_name = 'Observation vector Y'
        obs_vector.comment1 = 'mixed units -- see obs_flag field above'

        obs_vector_uncertainty = fid.createVariable('obs_vector_uncertainty', 'f4', ('time','obs_dim',))
        obs_vector_uncertainty.long_name = '1-sigma uncertainty in the observation vector (sigY)'
        obs_vector_uncertainty.comment1 = 'mixed units -- see obs_flag field above'

        forward_calc = fid.createVariable('forward_calc', 'f4', ('time','obs_dim',))
        forward_calc.long_name = 'Forward calculation from state vector (i.e., F(Xn))'
        forward_calc.comment1 = 'mixed units -- see obs_flag field above'

        # If we are trying to keep the output file small, then do not include
        # these fields in the output file
        if vip['output_file_keep_small'] == 0:
            arb1 = fid.createVariable('arb1', 'i2', ('arb_dim1',))
            arb1.long_name = 'Arbitrary dimension'
            arb1.comment = ('contains (1) temperature profile, (2) water vapor profile'
                       + ' (3) liquid water path, (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide'
                       + ' (8) methane, (9) nitrous oxide')

            arb2 = fid.createVariable('arb2', 'i2', ('arb_dim2',))
            arb2.long_name = 'Arbitrary dimension'
            arb2.comment = ('contains (1) temperature profile, (2) water vapor profile'
                       + ' (3) liquid water path , (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide' 
                       + ' (8) methane, (9) nitrous oxide')

            Xop = fid.createVariable('Xop', 'f4', ('time','arb_dim1',))
            Xop.long_name = 'Optimal solution'
            Xop.comment1 = 'mixed units -- see field arb above'

            Sop = fid.createVariable('Sop', 'f4', ('time','arb_dim1','arb_dim2',))
            Sop.long_name = 'Covariance matrix of the solution'
            Sop.comment1 = 'mixed units -- see field arb above'

            Akernal = fid.createVariable('Akernal', 'f4', ('time','arb_dim1','arb_dim2',))
            Akernal.long_name = 'Averaging kernal'
            Akernal.comment1 = 'mixed units -- see field arb above'

            Xa = fid.createVariable('Xa', 'f4', ('arb_dim1',))
            Xa.long_name = 'Prior mean state'
            Xa.comment1 = 'mixed units -- see field arb above'

            Sa = fid.createVariable('Sa', 'f4', ('arb_dim1','arb_dim2',))
            Sa.long_name = 'Prior covariance'
            Sa.comment1 = 'mixed units -- see field arb above'

        # These should be the last three variables in the file
        lat = fid.createVariable('lat', 'f4')
        lat.long_name = 'Latitude'
        lat.units = 'degrees_north'

        lon = fid.createVariable('lon', 'f4')
        lon.long_name = 'Longitude'
        lon.units = 'degrees_east'

        alt = fid.createVariable('alt', 'f4')
        alt.long_name = 'station height above mean sea level'
        alt.units = 'm'

        # Add some global attributes
        for i in range(len(list(globatt.keys()))):
            fid.setncattr(list(globatt.keys())[i], globatt[list(globatt.keys())[i]])
        fid.Prior_dataset_comment = prior['comment']
        fid.Prior_dataset_filename = prior['filename']
        fid.Prior_dataset_number_profiles = prior['nsonde']
        fid.Prior_dataset_T_inflation_factor = str(vip['prior_t_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_t_iht']) + ' km AGL'
        fid.Prior_dataset_Q_inflation_factor = str(vip['prior_q_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_q_iht']) + ' km AGL'
        fid.Prior_dataset_TQ_correlation_reduction_factor = vip['prior_tq_cov_val']
        fid.Total_clock_execution_time_in_s = exectime
        fid.Retrieval_option_flags = '{:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}'.format(modeflag[0], modeflag[1], modeflag[2], modeflag[3], modeflag[4], modeflag[5], modeflag[6], modeflag[7], modeflag[8])
        fid.vip_tres = (str(vip['tres']) + ' minutes. Note that the sample time corresponds to the '
                      + 'center of the averaging interval. A value of 0 implies that no averaging was performed')
        fid.Retrieval_start_hour = shour
        fid.Conventions = 'CF-1.10'
        add_vip_to_global_atts(fid, vip)

        # Add some of the static (non-time-dependent) data
        base_time[:] = xret[0]['secs']
        height[:] = xret[0]['z']
        obs_flag[:] = xret[0]['flagY']
        obs_dimension[:] = xret[0]['dimY']

        ones = np.ones(nht)
        twos = np.ones(nht)*2
        tmp = np.append(ones,twos)
        tmp = np.append(tmp, np.array([3,4,5,6,7,7,7,8,8,8,9,9,9]))

        if vip['output_file_keep_small'] == 0:
            arb1[:] = tmp
            arb2[:] = tmp
            Xa[:] = prior['Xa']
            Sa[:,:] = prior['Sa']

        # Fill these in, if available
        if type(location) is dict:
            if ((type(location['lat']) is int) or (type(location['lat'] is float))):
                try:
                    lat[:] = np.float(location['lat'])
                except TypeError:
                    # TODO - Allow changing lat/lon/alt. I.e. let the lat/lon/alt have a time dimension
                    lat[:] = np.float(np.mean(location['lat']))
            else:
                lat[:] = -999.0

            if ((type(location['lon']) is int) or (type(location['lon'] is float))):
                try:
                    lon[:] = np.float(location['lon'])
                except TypeError:
                    lon[:] = np.float(np.mean(location['lon']))
            else:
                lon[:] = -999.0

            if ((type(location['alt']) is int) or (type(location['alt'] is float))):
                try:
                    alt[:] = np.float(location['alt'])
                except TypeError:
                    alt[:] = np.float(np.mean(location['alt']))
            else:
                alt[:] = -999.0

        else:
            lat[:] = -999.
            lon[:] = -999.
            alt[:] = -999.

        fid.close()
        
    # Commenting this out but leaving it here in case we need to come back to it
    
    # foo = np.where(xret[0]['z'] < vip['prior_chimney_ht'])[0]
    # if len(foo) > 0:
    #     temp_tmp[foo,:] = nochim
    #     wvmr_tmp[foo,:] = nochim
    #     stemp_tmp[foo,:] = nochim
    #     swvmr_tmp[foo,:] = nochim
    #     theta_tmp[foo,:] = nochim
    #     thetae_tmp[foo,:] = nochim
    #     dewpt_tmp[foo,:] = nochim
    #     rh[foo,:] = nochim

    # Now append all of the samples from fsample onward into the file
    if verbose >= 3:
        print('Appending data to ' + nfilename)
    fid = Dataset(nfilename, 'a')
    fid.Total_clock_execution_time_in_s = str(exectime)

    time_offset = fid.variables['time_offset']
    time = fid.variables['time']
    hour = fid.variables['hour']
    qc_flag = fid.variables['qc_flag']

    temperature = fid.variables['temperature']
    waterVapor = fid.variables['waterVapor']
    lwp = fid.variables['lwp']
    lReff = fid.variables['lReff']
    iTau = fid.variables['iTau']
    iReff = fid.variables['iReff']
    co2 = fid.variables['co2']
    ch4 = fid.variables['ch4']
    n2o = fid.variables['n2o']

    sigmaT = fid.variables['sigma_temperature']
    sigmaWV = fid.variables['sigma_waterVapor']
    sigma_lwp = fid.variables['sigma_lwp']
    sigma_lReff = fid.variables['sigma_lReff']
    sigma_iTau = fid.variables['sigma_iTau']
    sigma_iReff = fid.variables['sigma_iReff']
    sigma_co2 = fid.variables['sigma_co2']
    sigma_ch4 = fid.variables['sigma_ch4']
    sigma_n2o = fid.variables['sigma_n2o']

    converged_flag = fid.variables['converged_flag']
    gamma = fid.variables['gamma']
    n_iter = fid.variables['n_iter']
    rmsr = fid.variables['rmsr']
    rmsa = fid.variables['rmsa']
    rmsp = fid.variables['rmsp']
    chi2 = fid.variables['chi2']
    convergence_criteria = fid.variables['convergence_criteria']
    dfs = fid.variables['dfs']
    sic = fid.variables['sic']
    vres_temp = fid.variables['vres_temperature']
    vres_wv = fid.variables['vres_waterVapor']
    cdfs_temp = fid.variables['cdfs_temperature']
    cdfs_wv = fid.variables['cdfs_waterVapor']

    hatchOpen = fid.variables['hatchOpen']
    cbh = fid.variables['cbh']
    cbh_flag = fid.variables['cbh_flag']
    pressure = fid.variables['pressure']

    theta = fid.variables['theta']
    thetae = fid.variables['thetae']
    rh = fid.variables['rh']
    dewpt = fid.variables['dewpt']
    pwv = fid.variables['pwv']
    pblh = fid.variables['pblh']
    sbih = fid.variables['sbih']
    sbim = fid.variables['sbim']
    sblcl = fid.variables['sbLCL']
    sbcape = fid.variables['sbCAPE']
    sbcin = fid.variables['sbCIN']
    mllcl = fid.variables['mlLCL']
    mlcape = fid.variables['mlCAPE']
    mlcin = fid.variables['mlCIN']
    
    sigma_pwv = fid.variables['sigma_pwv']
    sigma_pblh = fid.variables['sigma_pblh']
    sigma_sbih = fid.variables['sigma_sbih']
    sigma_sbim = fid.variables['sigma_sbim']
    sigma_sblcl = fid.variables['sigma_sbLCL']
    sigma_sbcape = fid.variables['sigma_sbCAPE']
    sigma_sbcin = fid.variables['sigma_sbCIN']
    sigma_mllcl = fid.variables['sigma_mlLCL']
    sigma_mlcape = fid.variables['sigma_mlCAPE']
    sigma_mlcin = fid.variables['sigma_mlCIN']

    obs_vector = fid.variables['obs_vector']
    obs_vector_uncertainty = fid.variables['obs_vector_uncertainty']
    forward_calc = fid.variables['forward_calc']

    if vip['output_file_keep_small'] == 0:
        Xop = fid.variables['Xop']
        Sop = fid.variables['Sop']
        Akernal = fid.variables['Akernal']

    basetime = fid.variables['base_time'][:]
    
    time_offset[fsample] = xret[fsample]['secs'] - basetime
    time[fsample] = xret[fsample]['hour']*60*60     # compute the seconds since midnight
    hour[fsample] = xret[fsample]['hour']
    qc_flag[fsample] = xret[fsample]['qcflag']

    did = np.where(np.array(list(fid.dimensions.keys())) == 'height')[0]
    if len(did) == 0:
        print('Whoaa -- this should not happen -- aborting')
        return success, nfilename
    
    if fid.dimensions['height'].size != len(xret[0]['z']):
        print('Whoaa -- this should not happen size -- aborting')
        return success, nfilename

    temperature[fsample,:] = xret[fsample]['Xn'][0:nht]
    waterVapor[fsample,:] = xret[fsample]['Xn'][nht:2*nht]
    lwp[fsample] = xret[fsample]['Xn'][2*nht]
    lReff[fsample] = xret[fsample]['Xn'][2*nht+1]
    iTau[fsample] = xret[fsample]['Xn'][2*nht+2]
    iReff[fsample] = xret[fsample]['Xn'][2*nht+3]
    co2[fsample,:] = xret[fsample]['Xn'][2*nht+4:2*nht+7]
    ch4[fsample,:] = xret[fsample]['Xn'][2*nht+7:2*nht+10]
    n2o[fsample,:] = xret[fsample]['Xn'][2*nht+10:2*nht+13]

    sig = np.sqrt(np.diag(xret[fsample]['Sop']))
    sigmaT[fsample,:] = sig[:nht]
    sigmaWV[fsample,:] = sig[nht:2*nht]
    sigma_lwp[fsample] = sig[2*nht]
    sigma_lReff[fsample] = sig[2*nht+1]
    sigma_iTau[fsample] = sig[2*nht+2]
    sigma_iReff[fsample] = sig[2*nht+3]
    sigma_co2[fsample,:] = sig[2*nht+4:2*nht+7]
    sigma_ch4[fsample,:] = sig[2*nht+7:2*nht+10]
    sigma_n2o[fsample,:] = sig[2*nht+10:2*nht+13]

    converged_flag[fsample] = xret[fsample]['converged']
    gamma[fsample] = xret[fsample]['gamma']
    n_iter[fsample] = xret[fsample]['niter']
    rmsr[fsample] = xret[fsample]['rmsr']
    rmsa[fsample] = xret[fsample]['rmsa']
    rmsp[fsample] = xret[fsample]['rmsp']
    chi2[fsample] = xret[fsample]['chi2']
    convergence_criteria[fsample] = xret[fsample]['di2m']
    dfs[fsample,:] = xret[fsample]['dfs']
    sic[fsample] = xret[fsample]['sic']
    vres_temp[fsample,:] = xret[fsample]['vres'][0,:]
    vres_wv[fsample,:] = xret[fsample]['vres'][1,:]
    cdfs_temp[fsample,:] = xret[fsample]['cdfs'][0,:]
    cdfs_wv[fsample,:] = xret[fsample]['cdfs'][1,:]
    
    hatchOpen[fsample] = xret[fsample]['hatchopen']
    cbh[fsample] = xret[fsample]['cbh']
    cbh_flag[fsample] = xret[fsample]['cbhflag']
    pressure[fsample,:] = xret[fsample]['p'][0:nht]

    theta[fsample,:] = derived['theta'][:]
    thetae[fsample,:] = derived['thetae'][:]
    rh[fsample,:] = derived['rh'][:]
    dewpt[fsample,:] = derived['dewpt'][:]
    
    pwv[fsample] = dindex['indices'][0]
    pblh[fsample] = dindex['indices'][1]
    sbih[fsample] = dindex['indices'][2]
    sbim[fsample] = dindex['indices'][3]
    sblcl[fsample] = dindex['indices'][4]
    sbcape[fsample] = dindex['indices'][5]
    sbcin[fsample] = dindex['indices'][6]
    mllcl[fsample] = dindex['indices'][7]
    mlcape[fsample] = dindex['indices'][8]
    mlcin[fsample] = dindex['indices'][9]
    
    sigma_pwv[fsample] = dindex['sigma_indices'][0]
    sigma_pblh[fsample] = dindex['sigma_indices'][1]
    sigma_sbih[fsample] = dindex['sigma_indices'][2]
    sigma_sbim[fsample] = dindex['sigma_indices'][3]
    sigma_sblcl[fsample] = dindex['sigma_indices'][4]
    sigma_sbcape[fsample] = dindex['sigma_indices'][5]
    sigma_sbcin[fsample] = dindex['sigma_indices'][6]
    sigma_mllcl[fsample] = dindex['sigma_indices'][7]
    sigma_mlcape[fsample] = dindex['sigma_indices'][8]
    sigma_mlcin[fsample] = dindex['sigma_indices'][9]

    obs_vector[fsample,:] = xret[fsample]['Y']
    obs_vector_uncertainty[fsample,:] = xret[fsample]['sigY']
    forward_calc[fsample,:] = xret[fsample]['FXn']

    if vip['output_file_keep_small'] == 0:
        Xop[fsample,:] = xret[fsample]['Xn']
        Sop[fsample,:,:] = xret[fsample]['Sop']
        Akernal[fsample,:,:] = xret[fsample]['Akern']

    fid.close()
    success = 1

    return success, nfilename

################################################################################
# This function creates a "simulated" output structure needed for running the
# code in the "output_clobber eq 2" (append) mode. The number of fields, the
# dimensions and types of each of the fields, needs to be correct. Only the
# "secs" field needs to have the proper values though...
################################################################################

def create_xret(xret, fsample, vip, irs, Xa, Sa, z, bands, obsdim, obsflag,shour):

    # Find all of the output files with this date
    yy = np.array([datetime.utcfromtimestamp(x).year for x in irs['secs']])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in irs['secs']])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in irs['secs']])
    ymd = yy*10000 + mm*100 + dd

    files = []
    filename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + str(ymd[0]) + '.*.nc'
    files = files + (glob.glob(filename))

    # If none are found, then just run code as normal. Note that xret and fsample
    if len(files) == 0:
        if vip['output_clobber'] == 2:
            print('The flag output_clobber was set to 2 for append, but no prior file was found')
            print('    so code will run as normal')
            nfilename = ' '
            return xret, fsample, nfilename
        else:
            nfilename = ' '
            return xret, fsample, nfilename
    
    # Check to see if a file with the same shour exists
    found = False
    for i in range(len(files)):
        fid = Dataset(files[i],'r')
        previous_shour = fid.Retrieval_start_hour
        fid.close()
        if np.isclose(shour,previous_shour):
            found = True
            break

    if found and vip['output_clobber'] == 0:
        print('A file with the same rootname and shour as the current run was found. Aborting retrieval')
        print('     to prevent clobbering.')
        print('     Existing file name: ' + files[i])
        nfilename = files[i]
        return xret, -1, nfilename
    
    # There was a file for this day, but there were different shours
    if not found and vip['output_clobber'] == 0:
        nfilename = ' '
        return xret, fsample, nfilename
    
    if not found and vip['output_clobber'] == 2:
        print('The flag output_clobber was set to 2 for append, but no prior file')
        print('      with the same shour was found so code will run as normal')
        print('      and create a new file.')
        nfilename = ' '
        return xret, fsample, nfilename
        
    if not found:
        print('Error: There seems to be some condition that was unanticipated -- aborting')
        sys.exit()

    # Otherwise, let's initialize from the last file
    nfilename = files[i]
    fid = Dataset(nfilename, 'r')
    bt = fid.variables['base_time'][:]
    to = fid.variables['time_offset'][:]
    xobsdim = fid.variables['obs_dimension'][:]
    xobsflag = fid.variables['obs_flag'][:]
    xz = fid.variables['height'][:]
    xXa = fid.variables['Xa'][:]
    xSa = fid.variables['Sa'][:]
    fid.close()

    secs = bt+to

    # Set up some default values
    w0idx, nY = Other_functions.find_wnum_idx(irs['wnum'],bands)
    wnum = irs['wnum'][w0idx]
    nY = len(xobsflag)
    Y = np.zeros(nY)
    Sop = np.diag(np.ones(len(Xa)))
    Kij = np.zeros((nY,len(Xa)))
    Gain = np.zeros((len(Xa),nY))
    Akern = np.copy(Sop)
    vres = np.zeros((2,len(z)))
    cdfs = np.zeros((2,len(z)))
    dfs = np.zeros(16)
    sic = 0.0

    # A few very basic checks to make sure that some of the variables in the
    # output file match the current ones (e.g., height, wavenumbers, etc)
    diff = np.abs(z-xz)
    foo = np.where(diff > 0.001)[0]
    if ((len(xz) != len(z)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in heights')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(Xa-xXa)

    foo = np.where(diff > 0.001)[0]
    if ((len(Xa) != len(xXa)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in Xa')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(obsdim - xobsdim)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsdim) != len(xobsdim)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_dim')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(obsflag - xobsflag)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsflag) != len(xobsflag)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_flag')
        fsample = -1
        return xret, fsample, nfilename

    # This structure must match that at the end of the main iteration loop
    # where the retrieval is performed. The values can be anything, with
    # the exception of the "secs" field

    xtmp = {'idx':0, 'secs':0.0, 'ymd':ymd[0], 'hour':0.0,
             'nX':len(Xa), 'nY':len(obsflag),
             'dimY':np.zeros(len(obsflag)),
             'Y':np.zeros(len(obsflag)),
             'sigY':np.zeros(len(obsflag)),
             'flagY':np.zeros(len(obsflag)), 'nitr':0,
             'z':z, 'p':np.zeros(len(z)), 'hatchopen':0,
             'cbh':0., 'cbhflag':0,
             'x0':Xa*0, 'Xn':Xa*0, 'Fxn':Y*0, 'Sop':Sop, 'K':Kij, 'Gain':Gain, 'Akern':Akern,
             'vres':vres, 'cdfs':cdfs, 'gamma':0., 'qcflag':0, 'sic':sic, 'dfs':dfs, 'di2m':0.,
             'rmsa':0., 'rmsr':0., 'rmsp':0., 'chi2':0., 'converged':0}

    xret = []
    xret.append(xtmp)
    xret[0]['secs'] = secs[0]
    for i in range(1,len(secs)):
        xret.append(xtmp)
        xret[i]['secs'] = secs[i]
    fsample = len(secs)

    return xret, fsample, nfilename
