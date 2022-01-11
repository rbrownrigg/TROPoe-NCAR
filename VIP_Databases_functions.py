import os
import shutil
import numpy as np
import scipy.io

################################################################################
# This file contains the following functions:
# read_vip_file()
# check_vip_file()
# abort()
# read_scat_databases()
# read_stdatmos()
################################################################################

# TODO -- add function that write out default VIP file to stdout and exits

################################################################################
#This routine reads in the controlling parameters from the VIP file.
################################################################################

def read_vip_file(filename,globatt,verbose,debug,dostop):

    maxbands = 200 #The maximum number of bands to enable for the retrieval

    # This is the stucture with all of the input that we need.
    # The code below will be searching the VIP input file for the same keys
    # that are listed in this structure. If a line is found with the same
    # name , then the code will determine the type for the value
    # (e.g., is the value for that key a float, string, integer, etc) and then
    # will cast the value provided in the VIP file as that.
    # There are three exceptions to this
    #   1) "Success" has the status of this routine -- it is not set by VIP file
    #   2) "aeri_calib_pres" is a 2-elements floating point array
    #   3) "vip_filename" captures the name of the input VIP file itself
    # The code will ouput how many of the keys in this structure were found.
    # Note that not all of them have to be in the VIP file; if a key in this
    # structure is not found in the VIP file then it maintains it default value

    vip = ({'success':0,
      'tres':0,                     #Temporal resolution [min], 0 implies native AERI temporal resolution
      'avg_instant':0,              #A flag to specify if this is an average (0) over the tres period, or instantaneous (1) sample (i.e. do not average the data)
      'tag':'tag',                  #String for temporary files / directories

      'aeri_type':0,                # 0- output, options, and stop, 1 - ARM AERI data, 2 - dmv2cdf AERI data (C1_rnc.cdf and _sum.cdf), 3 - dmv2ncdf (C1.RNC.cdf and .SUM.cdf) , -1 - MWR data is to be used as MASTER dataset (no AERI data being read in)
      'aeri_pca_nf':1,              # 0 - AERI data was NOT PCA noise filtered, 1 - AERI data was PCA noise filtered
      'aerich1_path':'None',          # Path to the AERI ch1 radiance files
      'aerisum_path':'None',          # Path to the AERI summary files
      'aerieng_path':'None',          # Path to the AERI engineering files
      'aeri_smooth_noise':0,        # The temporal window [minutes] used to smooth the AERI noise with time
      'aeri_calib_pres': [0.0,1.0], # Intercept [mb] and slope [mb/mb to calib (newP = int = slope*obsP) (need comma between them)
      'aeri_use_missingDataFlag':1, # Set this to 1 to use the field "missingDataFlag" (from the ch1 file) to remove bad AERI data from analysis. If not zero, then all AERI data will be processed,
      'aeri_hatch_switch':1,        # 1 - only include hatchOpen=1 when averaging, 2 - include all AERI samples when averaging
      'aeri_fv':0.0,                # Apply a foreoptics obscuration correction
      'aeri_fa':0.0,                # Apply an aftoptics obscuration correction
      'aeri_old_ffov_halfangle':23.0, # Original Half angle [millirad] of the finite field of view of the instrument
      'aeri_new_ffov_halfangle':0.0,  # New half angle [millirad] of the finite field of view of the instrument (values <= 0 will result in no correction being applied)
      'aeri_spectral_cal_factor':0.0, # The spectral calibration stretch factor to apply to the AERI data
      'aeri_min_675_bt':263.,       # Minimum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen
      'aeri_max_675_bt':313.,       # Maximum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen
      'aeri_spec_cal_factor': 1.0,

      'station_lat':-999.,             # Station latitude [degN]; if negative get value from AERI data file
      'station_lon':-999.,             # Station longitude [degE]; if negative get value from AERI data file
      'station_alt':-999.,             # Station altitude [m MSL]; if negative get value from AERI data file
      'station_psfc_min':800.,         # Default minimum surface pressure [mb]
      'station_psfc_max':1030.,        # Default maximum surface pressure [mb]

      'ext_wv_prof_type':0,         # External WV profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlprofmr), 3-NCAR WV DIAL, 4-Model sounding
      'ext_wv_prof_path':'None',      # Path to the external profile of WV data
      'ext_wv_prof_minht':0.0,      # Minimum height to use the data from the external WV profiler [km AGL]
      'ext_wv_prof_maxht':10.0,     # Maximum height to use the data from the external WV profiler [km AGL]
      'ext_wv_noise_mult_val':[1.0,1,1.0], # 3-element comma delimited list with the multipliers to apply the noise profile of the external water vapor profile (must be > 1)
      'ext_wv_noise_mult_hts':[0.0,3,20],  # 3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]
      'ext_wv_add_rel_error':0.0,   # When using the RLID, I may want to include a relative error contribution to the uncertainty to account for calibration. This is a correlated error component, and thus effects the off-diagonal elements of the observation covariance matrix. Units are [%]
      'ext_wv_time_delta':1.0,      # Maximum amount of time from endpoints of external WV dataset to extrapolate [hours]

      'ext_temp_prof_type':0,       # External temperature profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlproftemp); 4-Model sounding
      'ext_temp_prof_path':'None',    # Path to external profile of temp data
      'ext_temp_prof_minht':0.0,    # Minimum height to use the data from the external temp profiler [km AGL]
      'ext_temp_prof_maxht':10.0,   # Maximum height to use the data from the external temp profiler [km AGL]
      'ext_temp_noise_mult':1.0,    # Multiplicative value to apply the noise profile of the external temperature profile (must be >= 0)
      'ext_temp_noise_adder_val':[0.0,0,0], # 3-element comma delimited list of additive values to apply the noise profile of the external temperature profile (must be >= 0)
      'ext_temp_noise_adder_hts':[0.0,3,20], # 3-element comma delimited list with the corresponding heights for the additive value [km AGL]
      'ext_temp_time_delta':1.0,    # Maximum amount of time from endpoints of external temp dataset to extrapolate [hours]

      'mod_wv_prof_type':0,         # NWP model WV profile source: 0-none; 4-Model sounding
      'mod_wv_prof_path':'None',      # Path to the model profile of WV data
      'mod_wv_prof_minht':0.0,      # Minimum height to use the data from the model WV profiler [km AGL]
      'mod_wv_prof_maxht':10.0,     # Maximum height to use the data from the model WV profiler [km AGL]
      'mod_wv_noise_mult_val':[1.0,1,1.0], # 3-element comma delimited list with the multipliers to apply the noise profile of the model water vapor profile (must be > 1)
      'mod_wv_noise_mult_hts':[0.0,3,20],  # 3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]
      'mod_wv_time_delta':1.0,      # Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]

      'mod_temp_prof_type':0,       # NWP model temperature profile source: 0-none; 4-Model sounding
      'mod_temp_prof_path':'None',    # Path to the model profile of temp data
      'mod_temp_prof_minht':0.0,    # Minimum height to use the data from the model temp profiler [km AGL]
      'mod_temp_prof_maxht':10.0,   # Maximum height to use the data from the model temp profiler [km AGL]
      'mod_temp_noise_mult':1.0,    # Multiplicative value to apply the noise profile of the model temperature profile (must be >= 0)
      'mod_temp_noise_adder_val':[0.0,0,0], # 3-element comma delimited list of additive values to apply the noise profile of the model temperature profile (must be >= 0)
      'mod_temp_noise_adder_hts':[0.0,3,20], # 3-element comma delimited list with the corresponding heights for the additive value [km AGL]
      'mod_temp_time_delta':1.0,    # Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]

      'ext_sfc_temp_type':0,        # External surface temperature met data type: 0-none, 1-ARM met data [degC], 2-NCAR ISFS data [degC], 3-CLAMPS MWR met data [degC]
      'ext_sfc_temp_npts':1,        # Number of surface temperature met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation
      'ext_sfc_temp_rep_error':0.0, # Representativeness error for the surface temperature measurement [degC], which is added to the typical assumed uncertainty of 0.5 degC
      'ext_sfc_wv_type':0,          # External surface water vapor met data type: 0-none, 1-ARM met data [g/kg], 2-NCAR ISFS data [g/kg], 3-CLAMPS MWR met data [g/kg]
      'ext_sfc_wv_npts':1,          # Number of surface water vapor met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation
      'ext_sfc_wv_mult_error':1.0,  # Multiplier for the error in the surface water vapor measurement.  This is applied BEFORE the "rep_error" value
      'ext_sfc_wv_rep_error':0.0,   # Representativeness error for the surface water vapor measurement [g/kg], which is added to the typical assumed uncertainty of 0.5 degC and 3%RH
      'ext_sfc_path':'None',          # Path to the external surface met data
      'ext_sfc_time_delta':0.2,     # Maximum amount of time from endpoints of external surface met dataset to extrapolate [hours]
      'ext_sfc_relative_height':0,  # Relative height of the met station to the AERI zenith port [m]; note if met station is below AERI port then the value should be negative
      'ext_sfc_p_type': 0,          # 0 - Use the internal AERI pressure sensor for psfc; 1-ARM met data, 2-NCAR ISFS data, 3-CLAMPS MWR met data

      'co2_sfc_type':0,             # External CO2 surface data type: 0-none, 1-DDT QC PGS data
      'co2_sfc_npts':1,             # Number of surface CO2 in-situ points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation
      'co2_sfc_rep_error':0.0,      # Representativeness error for the CO2 surface measurement [ppm], which is added to the uncertainty of the obs in the input file
      'co2_sfc_path':'None',          # Path to the external surface CO2 data
      'co2_sfc_relative_height':0,  # Relative height of the CO2 surface measurement to the AERI zenith port [m]; note if in-situ obs is below AERI port then the value should be negative
      'co2_sfc_time_delta':1.5,     # Maximum amount of time from endpoints of external CO2 in-situ dataset to extrapolate [hours]

      'mwr_type':0,                 # 0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array
      'mwr_path':'None',              # Path to the MWR data
      'mwr_rootname':'mwr',         # Rootname of the MWR data files
      'mwr_elev_field':'elev',      # Name of the scene mirror elevation field; use "none" if all data are zenith
      'mwr_n_tb_fields':0,          # Number of fields to read in
      'mwr_tb_field_names':'tbsky23,tbsky31', # Comma separated list of field names for the Tb fields
      'mwr_tb_field1_tbmax':100.,   # Maximum value [K] in the first Tb field, used for QC
      'mwr_tb_freqs': '23.8,31.4',   # Comma separated list of frequency [GHz] of MWR Tb fields
      'mwr_tb_noise':'0.3,0.3',     # Comma separated list of noise levels [K] in the MWR Tb fields
      'mwr_tb_bias':'0.0,0.0',      # Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations
      'mwr_tb_replicate':1,         # Number of times to replicate the obs -- this allows us to change the weight of the MWR data in the retrieval (useful for LWP)
      'mwr_pwv_field':'pwv',        # Name of the PWV field in the MWR file
      'mwr_pwv_scalar':1.,          # Scalar used to multiply the MWR PWV field to convert to units of [cm]
      'mwr_lwp_field':'lwp',        # Name of the LWP field in the MWR file
      'mwr_lwp_scalar':1.,          # Scalar used to multiply the MWR LWP field to convert to units of [g/m2]
      'mwr_time_delta':0.083,       # The maximum amount of time [hours] that the MWR zenith obs must be to the sampling time to be used

      'mwrscan_type':0,             # 0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array
      'mwrscan_path':'None',          # Path to the MWRscan data
      'mwrscan_rootname':'mwr',     # Rootname of the MWRscan data files
      'mwrscan_elev_field':'elev',  # Name of the scene mirror elevation field; this field must exist
      'mwrscan_n_tb_fields':0,      # Number of fields to read in
      'mwrscan_n_elevations':2,     # The number of elevations to use in retrieval (put zenith obs in "mwr_type")
      'mwrscan_elevations':'20,160', # The elevation angles to use in deg, where 90 is zenith.  The code will look for these obs within the averaging interval
      'mwrscan_tb_field1_tbmax':330., # Maximum value [K] in the first Tb field, used for QC
      'mwrscan_tb_field_names':'tbsky23,tbsky31', # Comma separated list of field names for the Tb fields
      'mwrscan_tb_freqs':'23.8,31.4', # Comma separated list of frequency [GHz] of MWR
      'mwrscan_tb_noise':'0.3,0.3',  # Comma separated list of noise levels [K] in the MWR Tb fields
      'mwrscan_tb_bias':'0.0,0.0',   # Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations
      'mwrscan_time_delta':0.25,     # The maximum amount of time [hours] that the elevation scan must be to the sampling time to be used.

      'rass_prof_type':0,           # 0 - none, 5 - RASS Tv field has units C (no other values work)
      'rass_prof_path':'None',      # Path to the RASS data
      'rass_prof_minht':0.0,        # Minimum height to use the data from the RASS [km AGL]
      'rass_prof_maxht':5.0,        # Maximum height to use the data from the RASS [km AGL]
      'rass_noise_adder_val':[0.0,0.0,0.0],  # 3-element comma delimited list of additive values to apply the noise profile of the RASS temperature profile (must be >= 0).
      'rass_noise_adder_hts':[0.0,3,20],     # 3-element comma delimited list with the corresponding heights for the additive value [km AGL]
      'rass_time_delta':2.5,                 # The maximum amount of time [hours] that the RASS sample must be to the sampling time to be used.

      'cbh_type':0,                  # 0 - output options and stop, 1 - VCEIL, 2 - Gregs ASOS CBH file, 3 - CLAMPS DLfp data, 4 - ARM dlprofwstats data
      'cbh_path':'None',               # Path to the CBH data
      'cbh_window_in':20,            # Inner temporal window (full-size) centered upon AERI time to look for cloud
      'cbh_window_out':180,          # Outer temporal window (full-size) centered upon AERI time to look for cloud
      'cbh_default_ht':2.0,          # Default CBH height [km AGL], if no CBH data found

      'output_rootname':'None',        # String with the rootname of the output file
      'output_path':'None',            # Path where the output file will be placed
      'output_clobber':0,            # 0 - do not clobber preexisting output files, 1 - clobber them, 2 - append to the last file of this day
      'output_file_keep_small':0,    # 0 - all fields written; 1 - keep output file small by not including Sop, Akern, others

      'lbl_home':'/home/tropoe/vip/src/lblrtm_v12.1/lblrtm',               # String with the LBL_HOME path (environment variable)
      'lbl_version':'v12.1',         # String with the version information on LBLRTM
      'lbl_temp_dir':'/tmp',         # Temporary working directory for the retrieval
      'lbl_std_atmos':6,             # Standard atmosphere to use in LBLRTM and MonoRTM calcs
      'path_std_atmos':'/home/tropoe/vip/src/input/std_atmosphere.idl', # The path to the IDL save file with the standard atmosphere info in it
      'lbl_tape3':'tape3.data',      # The TAPE3 file to use in the lblrtm calculation.  Needs to be in the directory lbl_home/hitran/
      'monortm_version':'v5.0',        # String with the version information on MonoRTM
      'monortm_wrapper':'/home/tropoe/vip/src/monortm_v5.0/wrapper/monortm_v5', # Turner wrapper to run MonoRTM
      'monortm_exec':'/home/tropoe/vip/src/monortm_v5.0/monortm/monortm_v5.0_linux_gnu_sgl', # AERs MonoRTM executable
      'monortm_spec':'/home/tropoe/vip/src/monortm_v5.0/monolnfl_v1.0/TAPE3.spectral_lines.dat.0_55.v5.0_veryfast', # MonoRTM spectral database

      'lblrtm_jac_option':3,         # 1 - LBLRTM Finite Diffs, 2 - 3calc method, 3 - deltaOD method
      'lblrtm_forward_threshold':0., # The upper LWP threshold [g/m2] to use LBLRTM vs. radxfer in forward calculation
      'lblrtm_jac_interpol_npts_wnum':10, # The number of points per wnum to use in the compute_jacobian_interpol() function
      'monortm_jac_option':2,        # 1 - MonoRTM Finite Diffs, 2 - 3calc method
      'jac_max_ht':8.0,             # Maximum height to compute the Jacobian [km AGL]
      'max_iterations':10,           # The maximum number of iterations to use
      'first_guess':0,               # 1 - use prior as FG, 2 - use lapse rate and 60% RH profile as FG, 3 - use previous sample as FG
      'superadiabatic_maxht':0.300,  # The maximum height a superadiabatic layer at the surface can have [km AGL]
      'spectral_bands':np.zeros((2,maxbands)), # An array of spectral bands to use
      'retrieve_temp':1,             # 0 - do not retrieve temp, 1 - do retrieve temp (default)
      'retrieve_wvmr':1,             # 0 - do not retrieve wvmr, 1 - do retrieve wvmr (default)
      'retrieve_co2':0,              # 0 - do not retrieve co2, 1 - do retrieve co2 (exponential model), 2 - do retrieve co2 (step model)
      'fix_co2_shape':0,             # (This option only works with retrieve_co2=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed
      'retrieve_ch4':0,              # 0 - do not retrieve ch4, 1 - do retrieve ch4 (exponential model), 2 - do retrieve co2 (step model)
      'fix_ch4_shape':0,             # (This option only works with retrieve_ch4=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed
      'retrieve_n2o':0,             # 0 - do not retrieve n2o, 1 - do retrieve n2o (exponential model), 2 - do retrieve co2 (step model)
      'fix_n2o_shape':0,            # (This option only works with retrieve_n2o=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed
      'retrieve_lcloud':1,           # 0 - do not retrieve liquid clouds, 1 - retrieve liquid cloud properties
      'retrieve_icloud':0,           # 0 - do not retrieve   ice  clouds, 1 - retrieve   ice  cloud properties
      'lcloud_ssp':'/home/tropoe/vip/src/input/ssp_db.mie_wat.gamma_sigma_0p100',  # SSP file for liquid cloud properties
      'icloud_ssp':'/home/tropoe/vip/src/input/ssp_db.mie_ice.gamma_sigma_0p100',  # SSP file for   ice  cloud properties
      'qc_rms_value':10.0,            # The RMS value between ((obs minus calc)/obs_uncert) spectra, with values less than this being "good".  In short, if the solution is within n-sigma of the observation (where "n" is given by this value, then the retrieval is good
      'prior_t_ival':1.0,            # The prior inflation factor (>= 1) to apply at the surface for temperature
      'prior_t_iht':1.0,             # The height [km AGL] where the inflation factor goes to 1 (linear) for temperature
      'prior_q_ival':1.0,            # The prior inflation factor (>= 1) to apply at the surface for water vapor mixing ratio
      'prior_q_iht':1.0,             # The height [km AGL] where the inflation factor goes to 1 (linear) for water vapor mixing ratio
      'prior_tq_cov_val':1.0,        # A multiplicative value (0 < val <= 1) that is used to decrease the covariance in the prior between temperature and WVMR at all heights
      'prior_chimney_ht':0.0,        # The height of any "chimney" [km AGL]; prior data below this height are totally decorrelated
      'prior_co2_mn':[-1.0, 5, -5],  # Mean co2 concentration [ppm] (see "retrieve_co2" above)
      'prior_co2_sd':[ 2.0, 15,  3], # 1-sigma uncertainty in co2 [ppm]
      'prior_ch4_mn':[1.793, 0, -5], # Mean ch4 concentration [ppm] (see "retrieve_ch4" above)
      'prior_ch4_sd':[0.0538, 0.0015,  3], # 1-sigma uncertainty in ch4 [ppm]
      'prior_n2o_mn':[0.310,  0, -5], # Mean n2o concentration [ppm] (see "retrieve_n2o" above)
      'prior_n2o_sd':[0.0093, 0.00026,  3], # 1-sigma uncertainty in n2o [ppm]
      'prior_lwp_mn':10.0,            # Mean LWP [g/m2]
      'prior_lwp_sd':50.0,            # 1-sigma uncertainty in LWP [g/m2]
      'prior_lReff_mn':8.0,           # Mean liquid Reff [microns]
      'prior_lReff_sd':4.0,           # 1-sigma uncertainty in liquid Reff [microns]
      'prior_itau_mn':1.0,            # Mean ice cloud optical depth (geometric limit)
      'prior_itau_sd':5.0,            # 1-sigma uncertainty in ice cloud optical depth
      'prior_iReff_mn':25.0,          # Mean ice cloud Reff [microns]
      'prior_iReff_sd':8.0,           # 1-sigma uncertainty in ice cloud Reff [Reff]
      'min_PBL_height':0.3,           # The minimum height of the planetary boundary layer (used for trace gases) [km AGL]
      'max_PBL_height':5.0,           # The maximum height of the planetary boundary layer (used for trace gases) [km AGL]
      'nudge_PBL_height':0.5,         # The temperature offset (nudge) added to the surface temp to find PBL height [C]
      'vip_filename':'None'}          # Just for tracability
      )

    # Read in the file all at once

    if os.path.exists(filename):

        if verbose >= 1:
            print('Reading the VIP file: ' + filename)

        try:
            # inputt = np.genfromtxt(filename, dtype=str,comments='#', usecols = (0,1,2))
            inputt = np.genfromtxt(filename, dtype=str, comments='#', delimiter='=', autostrip=True)
        except Exception as e:
            print('There was an problem reading the VIP file')
    else:
        print('The VIP file ' + filename + ' does not exist')
        return vip

    if len(inputt) == 0:
        print('There were no valid lines found in the VIP file')
        return vip

    # Look for obsolete tags, and abort if they are found (forcing user to update their VIP file)
    obsolete_tags = ['AERI_LAT','AERI_LON','AERI_ALT','PSFC_MIN','PSFC_MAX']
    obsolete_idx  = np.zeros_like(obsolete_tags, dtype=int)
    vip_keys = [k.upper() for k in vip.keys()]
    for i in range(len(obsolete_tags)):
        foo = np.where(obsolete_tags[i].upper() == vip_keys)[0]
        if(len(foo) > 0):
            obsolete_idx[i] = 1
    foo = np.where(obsolete_idx > 0)[0]
    if(len(foo) > 0):
        print('Error: there were obsolete tags in the VIP file:')
        for i in range(len(foo)):
            print('     '+obsolete_tags[foo[i]])
            return vip

    # Look for these tags

    nfound = 1
    for key in vip.keys():
        if key != 'success':
            nfound += 1
            if key == 'vip_filename':
                vip['vip_filename'] = filename
            else:
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('Error: There were multiple lines with the same key in VIP file: ' + key)
                    return vip

                elif len(foo) == 1:
                    if verbose == 3:
                        print('Loading the key ' + key)
                    if key == 'spectral_bands':
                        bands = vip['spectral_bands']-1
                        tmp = inputt[foo,1][0].split(',')

                        if len(tmp) >= maxbands:
                            print('Error: There were more spectral bands defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                            return vip

                        for j in range(len(tmp)):
                            feh = tmp[j].split('-')
                            if len(feh) != 2:
                                print('Error: Unable to properly decompose the spectral_bands key')
                                if dostop:
                                    wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                                return vip
                            bands[0,j] = float(feh[0])
                            bands[1,j] = float(feh[1])
                        vip['spectral_bands'] = bands

                    elif key == 'aeri_calib_pres':
                        feh = inputt[foo,1][0].split(',')
                        if len(feh) != 2:
                            print('Error: The key aeri_calib_pres in VIP file must be intercept, slope')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip
                        vip['aeri_calib_pres'][0] = float(feh[0])
                        vip['aeri_calib_pres'][1] = float(feh[1])
                        if vip['aeri_calib_pres'][1] <= 0.0001:
                            print('Error: The key aeri_calib_pres in VIP file must have positive slope')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip

                    elif ((key == 'ext_wv_noise_mult_val') |
                          (key == 'ext_wv_noise_mult_hts') |
                          (key == 'ext_temp_noise_adder_val') |
                          (key == 'ext_temp_noise_adder_hts') |
                          (key == 'mod_wv_noise_mult_val') |
                          (key == 'mod_wv_noise_mult_hts') |
                          (key == 'mod_temp_noise_adder_val') |
                          (key == 'mod_temp_noise_adder_hts') |
                          (key == 'rass_noise_adder_val') |
                          (key == 'rass_noise_adder_hts') |
                          (key == 'prior_co2_mn') |
                          (key == 'prior_co2_sd') |
                          (key == 'prior_ch4_mn') |
                          (key == 'prior_ch4_sd') |
                          (key == 'prior_n2o_mn') |
                          (key == 'prior_n2o_sd') ):

                        feh = inputt[foo,1][0].split(',')
                        if len(feh) != len(vip[key]):
                            print('Error: The key ' + key + ' in VIP file must be a ' + str(len(vip[key])) + ' element array')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip

                        vip[key][0] = float(feh[0])
                        vip[key][1] = float(feh[1])
                        vip[key][2] = float(feh[2])

                    else:
                        vip[key] = type(vip[key])(inputt[foo,1][0])
                else:
                    if verbose == 3:
                        print('UNABLE to find the key ' + key)
                    nfound -= 1

    if verbose == 3:
        print(vip)
    if verbose >= 2:
        print('There were ' + str(nfound) + ' entries found out of ' + str(len(list(vip.keys()))))

    # Now look for any global attributes that might have been entered in the file

    matching = [s for s in inputt[:,0] if "globatt" in s]

    if verbose >= 2:
        print('There were ' + str(len(matching)) + ' global attributes found')

    for i in range(len(matching)):
        foo = np.where(matching[i] == inputt[:,0])[0]
        globatt[matching[i][8:]] = inputt[foo,1][0]

    vip['success'] = 1

    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')
    return vip

################################################################################
# This code performs some QC on the entries in the VIP structure to ensure
# that they are within a valid range.  Not every entry is checked...
################################################################################

def check_vip(vip):
    flag = 0           # Default is that everything is ok

    if ((vip['output_clobber'] < 0) | (vip['output_clobber'] > 2)):
        print('Error: The output_clobber flag can only be set to 0, 1, or 2')
        flag = 1

    if ((vip['aeri_fv'] < 0.0) | (vip['aeri_fv'] > 0.03)):
        print('Error: The AERI fv is too small or too large')
        flag = 1

    if ((vip['aeri_fa'] < 0.0) | (vip['aeri_fa'] > 0.03)):
        print('Error: The AERI fa is too small or too large')
        flag = 1

    if vip['jac_max_ht'] <= 1.0:
        print('Error: The maximum height to compute the Jacobian is too small; please increase')
        flag = 1

    if ((vip['max_iterations'] < 0) | (vip['max_iterations'] > 25)):
        print('Error: The maximum number of iterations must be between 0 and 25')
        flag = 1

    if ((vip['lbl_std_atmos'] < 1) | (vip['lbl_std_atmos'] > 6)):
        print('Error: The LBLRTM standard atmosphere must be an integer between 1 and 6')
        flag = 1

    if ((vip['aeri_use_missingDataFlag'] < 0) | (vip['aeri_use_missingDataFlag'] > 1)):
        print('Error: The aeri_use_missingDataFlag must be either 0 or 1')
        flag = 1

    if ((vip['aeri_hatch_switch'] < 0) | (vip['aeri_hatch_switch'] > 1)):
        print('Error: The aeri_hatch_switch must be either 0 or 1')
        flag = 1

    return flag

################################################################################
# This routine is called when AERIoe aborts
################################################################################

def abort(lbltmpdir, date):

    if os.path.exists(lbltmpdir):
        shutil.rmtree(lbltmpdir)

    print('>>> AERI retrieval on ' + str(date) + ' FAILED and ABORTED <<<')
    print('--------------------------------------------------------------------')
    print(' ')

################################################################################
#   This routine reads in the scattering properties from databases that were
#   created by Mie code (newmie_iteration) and by Ping Yang's computations (that
#   database was compiled by write_database.pro in IceScatProperties), and other
#   similar scattering property databases.  Rectangular matrices are returned,
#   which can then be processed by the routine get_scat_properties.pro to extract
#   out the properties depending on wavelength and effective radius.  Note that
#   there are two Mie databases: one for ice particles and one for water particles.
################################################################################

def read_scat_databases(dbname):

    #See if the database actually exists

    if not os.path.exists(dbname):
        print('ERROR: Unable to find the scattering database')
        return [], 1

    #Number of header lines (before the phase function angles)
    nheader = 5

    #Number of columns of data (not including the phase function)
    ncols = 13

    #Open and read the single scattering property database
    print('Reading: ' + dbname)
    f = open(dbname, 'r')
    f.readline()
    f.readline()
    nlines = int(f.readline().split()[0])
    nphase = int(f.readline().split()[0])

    if nlines <= 0:
        print('ERROR: There were no datalines found in this database -- this should not occur')
        f.close()
        return [], 1

    if nphase <= 0:
        print('ERROR: The scattering phase function was not defined in this database')
        f.close()
        return [], 1
    f.close()

    pangle = np.genfromtxt(dbname ,skip_header = nheader, max_rows = 1)

    data = np.genfromtxt(dbname ,skip_header = nheader+2).T

    fields = (['wavelength [um]','wavenumber [cm-1]','effective radius',
              'extinction cross section', 'scattering cross section',
              'absorption cross section', 'single scatter albedo',
              'asymmetry parameter', 'Extinction efficiency',
              'Absorption efficiency', 'Scattering efficiency', 'Volume',
              'Projected area','Rest of the elements are phase function'])

    database = ({'dbname':dbname, 'ncols':ncols, 'nphase':nphase, 'pangle':pangle,
                'data':data, 'columns':fields})

    return database, 0

################################################################################
# This function reads in the standard atmosphere profiles
################################################################################

def read_stdatmos(filename, stdatmos, verbose):

    if not os.path.exists(filename):
        print('Error: Unable to find the IDL save file with the standard atmosphere information in it')
        return {'status':0}

    temp = scipy.io.readsav(filename, python_dict = True)
    idx = stdatmos-1
    if ((idx < 0) | (idx >= len(temp['name']))):
        print('Error: the standard atmosphere specified is out-of-range')
        return {'status':0}

    if verbose >= 1:
        print('Using standard atmosphere: ' + temp['name'][idx].decode())

    return {'status':1, 'z':temp['z'][idx,:], 'p':temp['p'][idx,:], 't':temp['t'][idx,:], 'w':temp['w'][idx,:], 'pwv':temp['pwv'][idx], 'name':temp['name'][idx]}
