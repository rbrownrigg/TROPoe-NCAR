__version__ = '0.2.87'

import os
import sys
import numpy as np
import shutil
import scipy.io
import copy
import warnings
from netCDF4 import Dataset
from datetime import datetime
from time import gmtime, strftime
from subprocess import Popen, PIPE
from argparse import ArgumentParser

import Other_functions
import VIP_Databases_functions
import Calcs_Conversions
import Data_reads
import Jacobian_Functions
import Output_Functions

# Check to see if we are just writing out a blank vip
if '--vip' in sys.argv:
    # Write out a default vip file
    print("Writing default vip file to console")

    if '--experimental' in sys.argv:
        Output_Functions.write_example_vip_file(console=True, experimental=True)
    else:
        Output_Functions.write_example_vip_file(console=True)

    sys.exit()

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name if the VIP file (string)")
parser.add_argument("prior_filename", help="Name of the prior input dataset (string)")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last AERI sample]")
parser.add_argument("--verbose",type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--doplot", action="store_true", help="If set, then create real-time display of retrievals")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")
parser.add_argument("--dostop",action="store_true", help="Set this to stop at the end before exiting")

args = parser.parse_args()

date = args.date
vip_filename = args.vip_filename
prior_filename = args.prior_filename
shour = args.shour
ehour = args.ehour
verbose = args.verbose
doplot = args.doplot
debug = args.debug
dostop = args.dostop

#Check for the TPSS environment variables. They will override anything that is
#passed in via the command line

tpss_date = os.getenv("tpss_date")
tpss_pfile = os.getenv("tpss_pfile")
tpss_vfile = os.getenv("tpss_vfile")
tpss_shour = os.getenv("tpss_shour")
tpss_ehour = os.getenv("tpss_ehour")

if tpss_date:
    date = int(tpss_date)
if tpss_vfile:
    vip_filename = tpss_vfile
if tpss_pfile:
    prior_filename = tpss_pfile
if tpss_shour:
    shour = float(tpss_shour)
if tpss_ehour:
    ehour = float(tpss_ehour)

#Check to see if any of these are set; if not, fall back to default values

if shour is None:
    shour = 0.
if ehour is None:
    ehour = -1.
if verbose is None:
    verbose = 1
if debug is None:
    debug = False
if dostop is None:
    dostop = False

# If doplot is True, need to import matplotlib functions
if doplot:
    import matplotlib.pyplot as plt

# Initialize
success = True

# We need the background shell to be the C-shell, as we will be spawning out
# a variety of commands that make this assumption. So we will do a
# quick check to find out if the C-shell exists on this system, and if so,
# set the SHELL to this.

if verbose == 3:
    print(' ')
    print(('The current shell is', os.getenv('SHELL')))
else:
    warnings.filterwarnings("ignore", category=UserWarning)

process = Popen('which csh', stdout = PIPE, stderr = PIPE, shell=True)
stdout, stderr = process.communicate()

if stdout.decode() == '':
    print('Error: Unable to find the C-shell command on this system')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    sys.exit()
else:
    SHELL = stdout[:-1].decode()

if verbose == 3:
    print(('The shell for all commands is', SHELL))

#Capture the version of this file
globatt = {'algorithm_code': 'TROPoe Retrieval Code',
           'algorithm_author': 'Dave Turner, Earth System Research Laboratory / NOAA dave.turner@noaa.gov',
           'algorithm_comment1': 'TROPoe is a physical-iterative algorithm that retrieves thermodynamic profiles from ' +
                                 'a wide range of ground-based remote sensors.  It was primarily designed to use either ' +
                                 'infrared spectrometers or microwave radiometers as the primary instrument, and include ' +
                                 'observations from other sources to improve the quality of the retrieved profiles',
           'algorithm_comment2': 'Original code was written in IDL and is described by the "AERIoe" papers listed below',
           'algorithm_comment3': 'Code was ported to python by Joshua Gebauer with contributions ' +
                                 'from Tyler Bell (both at the University of Oklahoma)',
           'algorithm_version': __version__,
           'algorithm_reference1': 'DD Turner and U Loehnert, Information Content and ' +
                    'Uncertanties in Thermodynamic Profiles and Liquid Cloud Properties ' +
                    'Retrieved from the Ground-Based Atmospheric Emitted Radiance ' +
                    'Interferometer (AERI), J Appl Met Clim, vol 53, pp 752-771, 2014 ' +
                    'doi:10.1175/JAMC-D-13-0126.1',
           'algorithm_reference2': 'DD Turner and WG Blumberg, Improvements to the AERIoe ' +
                    'thermodynamic profile retrieval algorithm. IEEE Selected Topics ' +
                    'Appl. Earth Obs. Remote Sens., 12, 1339-1354, doi:10.1109/JSTARS.2018.2874968',
           'datafile_created_on_date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
           'datafile_created_on_machine': os.uname()[-1]}


# Start the retrieval
print(' ')
print('------------------------------------------------------------------------')
print(('>>> Starting TROPoe retrieval for ' + str(date) + ' (from ' + str(shour) + ' to ' + str(ehour) + ' UTC) <<<'))

#Find the VIP file and read it

vip = VIP_Databases_functions.read_vip_file(vip_filename, globatt = globatt, debug = debug, verbose = verbose, dostop = dostop)

if vip['success'] != 1:
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

process = Popen('echo $$', stdout = PIPE, stderr = PIPE, shell=True, executable = SHELL)
stdout, stderr = process.communicate()

uniquekey = vip['tag'] + '.' + stdout[:-1].decode()

if debug:
    print('DDT: Saving the VIP and globatt structure into "vip.npy" -- for debugging')
    np.save('vip.npy', vip)

# Make sure that Paul van Delst's script "lblrun" is in the $LBL_HOME/bin
# directory, as this is used often. The assumption is that if it is there,
# that the rest of the LBLRTM distribution is set up to use it properly.

if not os.path.exists(vip['lbl_home'] + '/bin/lblrun'):
    print('Error: Unable to find the script "lblrun" in the "lbl_home"/bin directory')
    print('This is a critical component of the LBLRTM configuration - aborting')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Check if the prior data exists
if not os.path.exists(prior_filename):
    print(('Error: Unable to find the prior data file: ' + prior_filename))
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

print(('Using the prior file: ' + prior_filename))


# Make sure that the output directory exists
if not os.path.exists(vip['output_path']):
    print('Error: The output directory does not exist')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Look at the name of the LBLtmpDir; if it starts with a "$"
# then assume that first part is an environment variable and
# decode the path accordingly

if vip['lbl_temp_dir'][0] == '$':
    envpath = vip['lbl_temp_dir'].split('/')
    tmpdir = os.getenv(envpath[0])
    if not tmpdir:
        print('Error: The LBLRTM temporary directory is being set to an environment variable that does not exist')
        print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
        print('--------------------------------------------------------------------')
        print(' ')
        sys.exit()
    for i in range(1,len(envpath)):
        tmpdir = tmpdir + '/' + envpath[i]
    vip['lbl_temp_dir'] = tmpdir

# Create the temporary working directory
lbltmpdir = vip['lbl_temp_dir'] + '/' + uniquekey
print(('Setting the temporary directory for RT model runs to: ' + lbltmpdir))

#Address this in Python 3 version
try:
    os.makedirs(lbltmpdir)
except:
    print('Error making the temporary directory')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')

#Now we are ready to start the main retrieval.
notqcov = 0    # I will leave this here for now, but later will add this to the vip file. If "1" it assumes no covariance between T & Q
cvgmult = 0.25 # I will leave this here for now, but later will add this to the vip file. It is a multiplier to apply to the convergence test (0.1 - 1.0)

success = 0

# House keeping stuff
starttime = datetime.now()
endtime = starttime
print(' ')

# Read in the SSP databases
sspl, flag = VIP_Databases_functions.read_scat_databases(vip['lcloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for liquid cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()


sspi, flag = VIP_Databases_functions.read_scat_databases(vip['icloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for ice cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Determine the minimum and maximum Reff values in these, but I need to have the
# minimum value be just a touch larger than the actual minimum value in the database
# hence the 1.01 multipliers

minLReff = np.nanmin(sspl['data'][2,:])*1.01
maxLReff = np.nanmax(sspl['data'][2,:])
miniReff = np.nanmin(sspi['data'][2,:])*1.01
maxiReff = np.nanmax(sspi['data'][2,:])

# Perform some more baseline checking of the keywords
if ((cvgmult  < 0.1) | (cvgmult > 1)):
    print('Error: cvgmult is too small or too large')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Perform some baseline checking of parameters in the VIP structure
# to make sure that the values are within the valid range
if VIP_Databases_functions.check_vip(vip) == 1:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Set switches associated with retrieving various variables and computing Jacobians
# This is a mess. Original was terribly inefficient. Maybe look to handle this better in future


if ((vip['retrieve_co2'] >= 1) & (vip['retrieve_co2'] <= 2)):
    doco2 = vip['retrieve_co2']
else:
    doco2 = 0
if ((vip['retrieve_ch4'] >= 1) & (vip['retrieve_ch4'] <= 2)):
    doch4 = vip['retrieve_ch4']
else:
    doch4 = 0
if ((vip['retrieve_n2o'] >= 1) & (vip['retrieve_n2o'] <= 2)):
    don2o = vip['retrieve_n2o']
else:
    don2o = 0
if vip['retrieve_lcloud'] >= 1:
    dolcloud = 1
    fixlcloud = 0           # Jacobian flag (for some reason 0 is on)
else:
    dolcloud = 0
    fixlcloud = 1           # Jacobian flag (for some reason 1 is off)
if vip['retrieve_icloud'] >= 1:
    doicloud = 1
    fixicloud = 0          # Jacobian flag (for some reason 0 is on)
else:
    doicloud = 0
    fixicloud = 1          # Jacobian flag (for some reason 1 is on)
if vip['retrieve_temp'] >= 1:
    dotemp = 1
    fixtemp = 0
else:
    dotemp = 0
    fixtemp = 1
if vip['retrieve_wvmr'] >= 1:
    dowvmr = 1
    fixwvmr = 0
else:
    dowvmr = 0
    fixwvmr = 1

modeflag = [dotemp, dowvmr, dolcloud, dolcloud, doicloud, doicloud, doco2, doch4, don2o]

# Select the LBLRTM version to use
print(' ')
print(('Working with the LBLRTM version ' + vip['lbl_version']))
print(('  in the directory ' + vip['lbl_home']))
print(('  and the TAPE3 file ' + vip['lbl_tape3']))
print(' ')

# Quick check: make sure the LBLRTM path is properly set
if not os.path.exists(vip['lbl_home'] + '/bin/lblrtm'):
    print('Error: lblhome is not properly set')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Make sure that the specified TAPE3 file exists
if not os.path.exists(vip['lbl_home'] + '/hitran/' + vip['lbl_tape3']):
    print('Error: unable to find the specified TAPE3 file in the LBL_HOME hitran directory')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Define some paths and constants
lbldir = lbltmpdir + '/lblout'       # Name of the lbl output directory
lbltp5 = lbltmpdir + '/lbltp5'       # Name of the tape5 file
lbltp3 = vip['lbl_tape3']           # Name of the tape3 file
lbllog = lbltmpdir + '/lbllog'       # Name of the lbl log file
lbltmp = lbltmpdir + '/lbltmp'       # Temporary directory for LBLRUN (will be LBL_RUN_ROOT)
monortm_config = 'monortm_config.txt'
monortm_zfreq = 'monortm_zfreqs.txt'    # For MWR-zenith calculations
monortm_sfreq = 'monortm_sfreqs.txt'    # For MWR-scan calculations
monortm_tfile = 'monortm_sonde.cdf'

# Make two commands: one for MWR-zenith and one for MWR-scan
monortm_zexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_zfreq + ' ; ' + vip['monortm_wrapper'])

monortm_sexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_sfreq + ' ; ' + vip['monortm_wrapper'])

# This should be included in the VIP file. Right now it is always set.
create_monortm_config = 1           # Set this flag to create a custom config file for MonoRTM
create_monortm_zfreq = 1            # Set this flag to create a custom freq-zenith file for MonoRTM
create_monortm_sfreq = 1            # Set this flag to create a custom freq-scan file for MonoRTM

#Load the standard atmosphere

stdatmos = VIP_Databases_functions.read_stdatmos(vip['path_std_atmos'], vip['lbl_std_atmos'], verbose)
if stdatmos['status'] == 0:
    print('Error: Unable to find/read the standard atmosphere file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Define the spectral regions(s) to use in the retrieval
bands = vip['spectral_bands']
foo = np.where(bands[0,:] >= 0)[0]
if len(foo) <= 0:
    print('Error: the spectral bands do not have any properly defined values')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
bands = bands[:,foo]

# Make sure that the MWR flags make sense
if vip['mwr_type'] > 0:
    if vip['mwr_tb_replicate'] <= 0:
        print('Error: when using MWR data the "replicate" flag must be >= 1')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

# Echo to the user the type of retrieval being performed
print(' ')
tmp = 'Retrieving: '
if dotemp == 1:
    tmp = tmp + 'T '
if dowvmr == 1:
    tmp = tmp + 'Q '
if dolcloud == 1:
    tmp = tmp + 'Liq_Cloud '
if doicloud == 1:
    tmp = tmp + 'Ice_Cloud '
if doco2 == 1:
    tmp = tmp + 'CO2 '
if doch4 == 1:
    tmp = tmp + 'CH4 '
if don2o == 1:
    tmp = tmp + 'N2O '
print(tmp)
print(' ')

# Read in the a priori covariance matrix of T/Q for this study
nsonde_prior = -1
try:
    fid = Dataset(prior_filename,'r')
except:
    print('Error: Unable to open the XaSa file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

z = fid.variables['height'][:]
Xa = fid.variables['mean_prior'][:]
Sa = fid.variables['covariance_prior'][:]
try:
    nsonde_prior = int(fid.Nsonde.split()[0])
except AttributeError:
    nsonde_prior = int(fid.Nprofiles.split()[0])

comment_prior = str(fid.Comment)
minT = float(fid.QC_limits_T.split()[5])
maxT = float(fid.QC_limits_T.split()[7])
if verbose == 3:
    print(('QC limits for T are ' + str(minT) + ' and ' + str(maxT)))
minQ = float(fid.QC_limits_q.split()[7])
maxQ = float( fid.QC_limits_q.split()[9])
if verbose == 3:
    print(('QC limits for Q are ' + str(minQ) + ' and ' + str(maxQ)))
fid.close()

if verbose >= 1:
    print(('Retrieved profiles will have ' + str(len(z)) + ' levels (from prior)'))
if verbose >= 2:
    print(('There were ' + str(nsonde_prior) + ' radiosondes used in the calculation of the prior'))

# Inflate the lowest levels of the prior covariance matrix, if desired

Sa, status = Other_functions.inflate_prior_covariance(Sa, z, vip['prior_t_ival'], vip['prior_t_iht'],
             vip['prior_q_ival'], vip['prior_q_iht'], vip['prior_tq_cov_val'],
             vip['prior_chimney_ht'], verbose)

if status == 0:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Get the bias spectrum
# Set this to an IDL (py) file that contains the bias to remove from the AERI data.
# Must contain at least two fields: "wnum" and "bias", where the latter is
# computed as AERI - LBLRTM. If this is not set, then no bias will be removed from
# the AERI data.  !!!!!NOTE!!!! THIS IS NOT CURRENTLY IMPLEMENTED.
remove_bias = 0
#if len(aeri_bias) > 0:
#    if not os.path.exists(aeri_bias):
#        print 'Error: The aeri_bias keyword is not set to a valid filename'
#        abort(lbltmpdir,date)
#        sys.exit()
#    else:
#        temp = scipy.io.readsav(aeri_bias, python_dict = True)
#    if ((len(temp['wnum']) == 0) | (len(temp['bias']) == 0)):
#        print 'Error: The fields "wnum" and/or "bias" do not exist in aeri_bias file'
#        abort(lbltmpdir,date)
#        sys.exit()
#    if len(temp['wnum']) != len(temp['bias']):
#        print 'Error: The fields "wnum" and "bias" do not have proper dimensions in rhe aeri_bias file'
#        abort(lbltmpdir,date)
#        sys.exit()
#    bwnum = temp['wnum']
#    bspec = temp['bias']
#    remove_bias = 1
#else:
#remove_bias = 0

# Read in the data
fail, aeri, mwr, mwrscan = Data_reads.read_all_data(date, z, vip['tres'], dostop, verbose, vip['avg_instant'],
    vip['aerich1_path'], vip['aeri_pca_nf'], vip['aeri_fv'], vip['aeri_fa'],
    vip['aerisum_path'], vip['aerieng_path'], vip['aeri_type'], vip['aeri_calib_pres'],
    vip['aeri_smooth_noise'], vip['aeri_use_missingDataFlag'],
    vip['aeri_min_675_bt'], vip['aeri_max_675_bt'], vip['aeri_spec_cal_factor'],
    vip['mwr_path'], vip['mwr_rootname'], vip['mwr_type'], vip['mwr_elev_field'], vip['mwr_n_tb_fields'],
    vip['mwr_tb_replicate'], vip['mwr_tb_field_names'], vip['mwr_tb_freqs'], vip['mwr_tb_noise'],
    vip['mwr_tb_bias'], vip['mwr_tb_field1_tbmax'], vip['cbh_path'], vip['cbh_type'], vip['cbh_window_in'],
    vip['cbh_window_out'], vip['cbh_default_ht'], vip['aeri_hatch_switch'],
    vip['aeri_use_missingDataFlag'], vip)

if fail == 1:
    print('Error reading in data: aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in any external sources of WV and temperature profiles

ext_prof = Data_reads.read_external_profile_data(date, z, aeri['secs'], vip['tres'], vip['avg_instant'],
              vip['ext_wv_prof_type'], vip['ext_wv_prof_path'], vip['ext_wv_noise_mult_hts'],
              vip['ext_wv_noise_mult_val'], vip['ext_wv_prof_minht'], vip['ext_wv_prof_maxht'],
              vip['ext_wv_time_delta'], vip['ext_temp_prof_type'], vip['ext_temp_prof_path'],
              vip['ext_temp_noise_adder_hts'], vip['ext_temp_noise_adder_val'],
              vip['ext_temp_prof_minht'], vip['ext_temp_prof_maxht'], vip['ext_temp_time_delta'],
              dostop, verbose)

if ext_prof['success'] != 1:
    print('Error: there is some problem in the external profile WV or temperature specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in any model sources of WV and temperature profiles

mod_prof = Data_reads.read_external_profile_data(date, z, aeri['secs'], vip['tres'], vip['avg_instant'],
              vip['mod_wv_prof_type'], vip['mod_wv_prof_path'], vip['mod_wv_noise_mult_hts'],
              vip['mod_wv_noise_mult_val'], vip['mod_wv_prof_minht'], vip['mod_wv_prof_maxht'],
              vip['mod_wv_time_delta'], vip['mod_temp_prof_type'], vip['mod_temp_prof_path'],
              vip['mod_temp_noise_adder_hts'], vip['mod_temp_noise_adder_val'],
              vip['mod_temp_prof_minht'], vip['mod_temp_prof_maxht'], vip['mod_temp_time_delta'],
              dostop, verbose)

if mod_prof['success'] != 1:
    print('Error: there is some problem in the model profile WV or temperature specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in any external sources of virtual temperature profiles (RASS)
    # I will need to tweak this call a bit by turning off the WV part of the call
if((vip['rass_prof_type'] == 0) | (vip['rass_prof_type'] == 5)):
    rass_prof = Data_reads.read_external_profile_data(date, z, aeri['secs'], vip['tres'], vip['avg_instant'],
              0, '.', vip['ext_wv_noise_mult_hts'],
              vip['ext_wv_noise_mult_val'], vip['ext_wv_prof_minht'], vip['ext_wv_prof_maxht'],
              vip['ext_wv_time_delta'], vip['rass_prof_type'], vip['rass_prof_path'],
              vip['rass_noise_adder_hts'], vip['rass_noise_adder_val'],
              vip['rass_prof_minht'], vip['rass_prof_maxht'], vip['rass_time_delta'],
              dostop, verbose)
else:
    print('Error: the RASS data weere assigned an invalid type in the VIP file (must be 0 or 5)')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
if(rass_prof['success'] != 1):
    print('  Warning: there is some problem in the RASS temperature specification')

# Read in any external time series data that would be used
ext_tseries = Data_reads.read_external_timeseries(date, aeri['secs'], vip['tres'], vip['avg_instant'],
              vip['ext_sfc_temp_type'], vip['ext_sfc_wv_type'], vip['ext_sfc_path'],
              vip['ext_sfc_temp_npts'], vip['ext_sfc_wv_npts'], vip['ext_sfc_temp_rep_error'],
              vip['ext_sfc_wv_mult_error'], vip['ext_sfc_wv_rep_error'], vip['ext_sfc_time_delta'],
              vip['ext_sfc_relative_height'], vip['co2_sfc_type'], vip['co2_sfc_npts'],
              vip['co2_sfc_rep_error'], vip['co2_sfc_path'], vip['co2_sfc_relative_height'],
              vip['co2_sfc_time_delta'], vip['ext_sfc_p_type'], dostop, verbose)

if ext_tseries['success'] != 1:
    print('Error: there is some problem in the external time series data specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# If this is the ASSIST data, then replace the surface pressure field (which is currently
# all missing values) with valid values for the retrieval.
if(vip['aeri_type'] == 5):
    if(vip['assist_pressure'] < 0):
        print('    Error: When using the ASSIST data as input, the keyword "assist_pressure" must be set in the VIP file')
        sys.exit()
    aeri['atmos_pres'][:] = vip['assist_pressure']

# If ehour < 0, then set it to the time of the last AERI sample. (This was needed
# for those cases when the AERI did not automatically reboot at 0 Z.)
if ehour < 0:
    ehour = np.nanmax(aeri['hour'])
    if verbose >= 2:
        print(('Resetting the processing end hour to ' + str(ehour) + ' UTC'))

# Capture the lat/lon/alt data in a structure
location = {'lat':aeri['lat'], 'lon':aeri['lon'], 'alt':aeri['alt']}
if vip['station_alt'] >= 0:
    if verbose >= 2:
        print('Overriding lat/lon/alt with info from VIP file')
    location = {'lat':vip['station_lat'], 'lon':vip['station_lon'], 'alt':vip['station_alt']}

# Very simple check to make sure station altitude makes sense [m MSL]
if(location['alt'] <= 0):
    print('    Error: the station altitude must be > 0 [m MSL]')
    sys.exit()

# Apply the AERI bias spectrum
if remove_bias == 1:
    #if verbose >= 2:
    #    print 'Removing a spectral bias from the AERI observations'
    #if ((np.nanmin(bwum)+1 < np.nanmin(aeri['wnum'])) | (np.nanmax(bwnum)-1 > np.nanmax(aeri['wnum']))):
    #    print 'Error: the spectral bias does not seem to cover the proper wnum range'
    #   VIP_Databases_functions.abort(lbltmpdir,date)
    #    sys.exit()
    #bias = np.interp(aeri['wnum'],bwnum,bspec)
    #for i in range(len(aeri['secs'])):
    #    aeri['radmn'][:,i] -= bias
    print('Error: Remove bias is not functional at all. Have to end.')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
else:
    if verbose >= 2:
        print('Did not remove a spectral bias from the AERI observations')

if notqcov == 1:
    k = len(z)
    for i in range(k):
        for j in range(k,2*k):
            Sa[i,j] = 0
            Sa[j,i] = 0

# Do I use a hardcoded value for CO2, or use my simple model to predict
# the concentration? Unit is ppm

if vip['prior_co2_mn'][0] < 0:
   vip['prior_co2_mn'][0] = Other_functions.predict_co2_concentration(aeri['yy'][0], aeri['mm'][0], aeri['dd'][0])

# Quick test to make sure that the trace gas models make sense. If not, abort
tmpco2 = Other_functions.trace_gas_prof(vip['retrieve_co2'], z, vip['prior_co2_mn'])
nfooco2 = len(np.where(tmpco2 < 0)[0])
tmpch4 = Other_functions.trace_gas_prof(vip['retrieve_ch4'], z, vip['prior_ch4_mn'])
nfooch4 = len(np.where(tmpch4 < 0)[0])
tmpn2o = Other_functions.trace_gas_prof(vip['retrieve_n2o'], z, vip['prior_n2o_mn'])
nfoon2o = len(np.where(tmpn2o < 0)[0])
if ((nfooco2 > 0) | (nfooch4 > 0) | (nfoon2o > 0)):
    print('Error: The CO2, CH4, and/or N2O parameters are incorrect giving negative values - aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Splice these trace gases and clouds into the Xa and Sa matrices.
# I am assuming no correlations between the TGs and clouds and the T/Q profiles
# The state vector is going to be defined to be:
# X = [T(z), q(z), LWP, ReL, tauI, ReI, CO2(3), CH4(3), N2O(3)
minsd = 1e-5        # Smallest value that doesn't result in problems in the matrix inversions below
diag = np.diag([np.max([vip['prior_lwp_sd'],minsd])**2,           # LWP : index 0
               np.max([vip['prior_lReff_sd'],minsd])**2,         # liquid Reff : index 1
               np.max([vip['prior_itau_sd'],minsd])**2,          # ice optical depth : index 2
               np.max([vip['prior_iReff_sd'],minsd])**2,         # ice Reff : index 3
               np.max([vip['prior_co2_sd'][0],minsd])**2,        #
               np.max([vip['prior_co2_sd'][1],minsd])**2,        # CO2 : indices 4,5,6
               np.max([vip['prior_co2_sd'][2],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][0],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][1],minsd])**2,        # CH4 : indices 7,8,9
               np.max([vip['prior_ch4_sd'][2],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][0],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][1],minsd])**2,        # N2O : indices 10,11,12
               np.max([vip['prior_n2o_sd'][2],minsd])**2])       #

zero = np.zeros((len(diag[0,:]),len(Xa)))
Sa = np.append(np.append(Sa,zero,axis=0),np.append(zero.T,diag,axis=0),axis=1)
Xa = np.append(Xa, [vip['prior_lwp_mn'], vip['prior_lReff_mn'],
               vip['prior_itau_mn'], vip['prior_iReff_mn'],
               vip['prior_co2_mn'][0], vip['prior_co2_mn'][1], vip['prior_co2_mn'][2],
               vip['prior_ch4_mn'][0], vip['prior_ch4_mn'][1], vip['prior_ch4_mn'][2],
               vip['prior_n2o_mn'][0], vip['prior_n2o_mn'][1], vip['prior_n2o_mn'][2]])
sig_Xa = np.sqrt(np.diag(Sa))

# Put all of the prior information into a structure for later
prior = {'comment':comment_prior, 'filename':prior_filename, 'nsonde':nsonde_prior,
         'Xa':np.copy(Xa), 'Sa':np.copy(Sa)}

# Make a directory for LBL_RUN_ROOT
if os.path.exists(lbltmp):
    shutil.rmtree(lbltmp)

os.mkdir(lbltmp)

# I want the code to try at least one retrieval, so I may need to
# modify the end time accordingly
foo = np.where((shour <= aeri['hour']) & (aeri['hour'] <= ehour))[0]
if len(foo) <= 0:
    foo = np.where(shour <= aeri['hour'])[0]
    if len(foo) <= 0:
        print('No samples were found after this start time. Quitting')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    print('Resetting the end hour to process at least 1 sample')
    if len(foo) < 2:
        ehour = aeri['hour'][foo] + 1/3600.     # Added 1 second to this AERI sample to make sure to get it
    else:
        ehour = aeri['hour'][foo[0]] + 1/3600.

# If neither liquid or ice clouds are enabled, then indicate that
# all retrievals are done as clear sky
if ((vip['retrieve_lcloud'] == 0) & (vip['retrieve_icloud'] == 0)):
    if verbose >= 2:
        print('All cloud retrievals disabled -- assuming clear sky')
    Xa[2*len(z)] = 0  # Zero LWP
    Xa[2*len(z)+2] = 0 # Zero ice optical depth
    aeri['cbhflag'][:] = 0        # Reset all flags to clear sky
    # Note that I am leaving the CBH values untouched...

# Now loop over the observations and perform the retrievals
xret = []                  #Initialize. Will overwrite this if a succesful ret
read_deltaod = 0           # Initialize.
already_saved = 0          # Flag to say if saved already or not...
fsample = 0                # Counter for number of spectra processed
cbh_string = ['Clear Sky', 'Inner Window', 'Outer Window', 'Default CBH']

# Quick check to make sure that the spectral bands being selected are actually
# valid for this interferometer (this ensures spectral range matches calculation below)
if(vip['aeri_type'] >= 1):
    minv = np.min(aeri['wnum'])
    maxv = np.max(aeri['wnum'])
    foo = np.where(bands < minv)
    if(len(foo) > 0):
        bands[foo] = minv+0.1
    foo = np.where(bands > maxv)
    if(len(foo) > 0):
        bands[foo] = maxv-0.1

# If clobber == 2, then we will try to append. But this requires that
# I perform a check to make sure that we are appending to a file that was
# created by a version of the code that makes sense. I only need to make this
# test once, hence this flag
if vip['output_clobber'] == 2:
    check_clobber = 1
else:
    check_clobber = 0

# This defines the extra vertical layers that will be added to both
# the infrared and microwave radiative transfer calculations
rt_extra_layers = Other_functions.compute_extra_layers(np.max(z))

version = ''
noutfilename = ''
################################################################################
# This is the main loop for the retrieval!
################################################################################
for i in range(len(aeri['secs'])):                        # { loop_i
    if ((shour >= aeri['hour'][i]) | (aeri['hour'][i] > ehour)):
        continue

    # Make sure that the AERI data aren't missing or considered bad
    if ((aeri['missingDataFlag'][i] > 0) | (aeri['missingDataFlag'][i] <= -1)):
        if(vip['aeri_type'] <= -1):
            itype = 'MWR'
        else:
            itype = 'AERI'
        print(f"  Sample {i:2d} at {aeri['hour'][i]:.4f} UTC -- no valid {itype:s} data found")
        continue
    else:
        print(f"  Sample {i:2d} at {aeri['hour'][i]:.4f} UTC is being processed (cbh is {aeri['cbh'][i]:.3f})")

    # See if we want to use the external sfc pressure instead of aeri pressure
    # and check to make sure external data read went okay
    if ((vip['ext_sfc_p_type'] > 0) & (ext_tseries['nPsfc'] >= 0)):
        print("Replacing AERI pressure with " +  ext_tseries['ptype'] + " pressure")
        aeri['atmos_pres'][i] = ext_tseries['psfc'][i]

    # Make sure the AERI's surface pressure is a valid value, as
    # this is needed to construct a pressure profile from the current X

    if ((vip['station_psfc_min'] > aeri['atmos_pres'][i]) | (aeri['atmos_pres'][i] > vip['station_psfc_max'])):
        print('Error: Surface pressure is not within range set in VIP -- skipping sample')
        continue

    # Select the spectral range to use for the retrieval
    # Define the observation vector and its covariance matrix
    # I'm going to try unapodized spectra, so the cov matrix is diagonal

    w0idx, nY = Other_functions.find_wnum_idx(aeri['wnum'],bands)
    if nY < 0:
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    wnum = np.copy(aeri['wnum'][w0idx])
    Y = np.copy(aeri['radmn'][w0idx,i])              # This is the actual observation vector
    sigY = np.copy(aeri['noise'][w0idx,i])           # This is the assumed 1-sigma uncertainty of the obs

    # I need a flag for the observations, so I can select the proper forward model.
    # The values are:
    #                1 -- AERI
    #                2 -- MWR zenith data
    #                3 -- external temperature profiler (sonde, lidar, NWP, etc) (*)
    #                4 -- external water vapor profiler (sonde, lidar, NWP, etc) (*)
    #                5 -- surface temperature measurement
    #                6 -- surface water vapor measurement
    #                7 -- external NWP model profile (*)
    #                8 -- external NWP model water vapor profile (*)
    #                9 -- in-situ surface CO2 obs
    #               10 -- MWR non-zenith data
    #               11 -- RASS virtural temperature data
    #               12, ... -- IASI, CrIS, S-HIS, others...
    #     (*) Note that the NWP input could come in two ways, but I wanted a way
    #         to bring in lidar and NWP data, while retaining backwards compatibility

    flagY = np.ones(wnum.shape)               # This will capture the source of input
    dimY = np.copy(wnum)                      # This will capture the dimension properly

    # Start building the observational covariance matrix. I am assuming that
    # the uncertainties in the AERI radiance are uncorrelated (channel-to-channel)
    # If I want to add off-diagonal elements (due to application of the noise
    # filter) this is the right place to do that....
    Sy = np.diag(sigY**2)

    # Append on the other observations, making sur to capture their "type"
    # using the flag values above and their dimensions. Note that I am assuming
    # that the covariance matrices between the different types of observations
    # are independent of each other, but in some cases I do allow off-diagonal
    # elements in the covariance matrix for that instrument

    if mwr['n_fields'] > 0:
        tbsky = np.copy(mwr['tbsky'][:,i])
        noise = np.copy(mwr['noise'])
        freq = np.copy(mwr['freq'])
        for gg in range(1,vip['mwr_tb_replicate']):
            # If I am replicating the measurement (to increase it's weight, then
            # I don't want to perfectly replicate it. I will perturb the observation
            # slightly (i.e., one-tenth of the noise level) -- this is what I am doing for surface met too.
            tbsky = np.append(tbsky, mwr['tbsky'][:,i] + np.random.normal(size = len(mwr['tbsky'][:,i]))*mwr['noise']/10.)
            noise = np.append(noise,mwr['noise'])
            freq = np.append(freq, mwr['freq'])

        foo = np.where(tbsky < 2.7)[0]                # Must be larger than the cosmic background
        if len(foo) > 0:
            tbsky[foo] = -999.
        Y = np.append(Y,tbsky)
        nSy = np.diag(noise**2)
        zero = np.zeros((len(noise),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY,noise)
        flagY = np.append(flagY, np.ones(vip['mwr_tb_replicate']*mwr['n_fields'])*2)
        dimY = np.append(dimY,freq)

    if ext_prof['nTprof'] > 0:
        foo = np.where((ext_prof['tempminht'] <= ext_prof['ht']) & (ext_prof['ht'] <= ext_prof['tempmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding external temp profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,ext_prof['temp'][foo,i])
        nSy = np.diag(ext_prof['sig_temp'][foo,i]**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, ext_prof['sig_temp'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*3)
        dimY = np.append(dimY, z[foo])

    if ext_prof['nQprof'] > 0:
        foo = np.where((ext_prof['wvminht'] <= ext_prof['ht']) & (ext_prof['ht'] <= ext_prof['wvmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding external wv profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,ext_prof['wv'][foo,i])
        if vip['ext_wv_add_rel_error'] > 0:
            if verbose >= 2:
                print('Adding systematic error to the external WV profile')
            nSy = Other_functions.add_sys_error(ext_prof['wv'][foo,i], ext_prof['sig_wv'][foo,i], vip['ext_wv_add_rel_error'])
        else:
            nSy = np.diag(ext_prof['sig_wv'][foo,i]**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, ext_prof['sig_wv'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*4)
        dimY = np.append(dimY, z[foo])

    if ext_tseries['nTsfc'] > 0:
        Y = np.append(Y, ext_tseries['temp'][:,i])
        nSy = np.diag(ext_tseries['stemp'][:,i]**2)
        zero = np.zeros((ext_tseries['nptsT'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, ext_tseries['stemp'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsT'])*5)
        dimY = np.append(dimY, np.ones(ext_tseries['nptsT'])*ext_tseries['sfc_relative_height'])

    if ext_tseries['nQsfc'] > 0:
        Y = np.append(Y, ext_tseries['wv'][:,i])
        nSy = np.diag(ext_tseries['swv'][:,i]**2)
        zero = np.zeros((ext_tseries['nptsQ'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, ext_tseries['swv'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsQ'])*6)
        dimY = np.append(dimY, np.ones(ext_tseries['nptsQ'])*ext_tseries['sfc_relative_height'])

    if mod_prof['nTprof'] > 0:
        foo = np.where((mod_prof['tempminht'] <= mod_prof['ht']) & (mod_prof['ht'] <= mod_prof['tempmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding model temp profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,mod_prof['temp'][foo,i])
        nSy = np.diag(mod_prof['sig_temp'][foo,i]**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, mod_prof['sig_temp'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*7)
        dimY = np.append(dimY, z[foo])

    if mod_prof['nQprof'] > 0:
        foo = np.where((mod_prof['wvminht'] <= mod_prof['ht']) & (mod_prof['ht'] <= mod_prof['wvmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding model wv profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,mod_prof['wv'][foo,i])
        nSy = np.diag(mod_prof['sig_wv'][foo,i])
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, mod_prof['sig_wv'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*8)
        dimY = np.append(dimY, z[foo])

    if ext_tseries['nCO2sfc'] > 0:
        Y = np.append(Y,ext_tseries['co2'][:,i])
        nSy = np.diag(ext_tseries['sco2'][:,i])
        zero = np.zeros((ext_tseries['nptsCO2'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, ext_tseries['sco2'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsCO2'])*9)
        dimY = np.append(dimY,np.ones(ext_tseries['nptsCO2'])*ext_tseries['co2_sfc_relative_height'])

    if mwrscan['n_fields'] > 0:
        tbsky = np.copy(mwrscan['tbsky'][:,i])
        noise = np.copy(mwrscan['noise'])
        foo = np.where(tbsky < 2.7)[0]
        if len(foo) > 0:
            tbsky[foo] = -999.
        Y = np.append(Y, tbsky)
        nSy = np.diag(noise**2)
        zero = np.zeros((len(noise),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, noise)
        flagY = np.append(flagY, np.ones(len(mwrscan['dim']))*10)
        dimY = np.append(dimY, mwrscan['dim'])

    if rass_prof['nTprof'] > 0:
        foo = np.where((rass_prof['tempminht'] <= rass_prof['ht']) & (rass_prof['ht'] <= rass_prof['tempmaxht']))[0]
        if(len(foo) == 0):
            print('  Error: the RASS heights were not found (and they should have been)')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,rass_prof['temp'][foo,i])
        nSy = np.diag(rass_prof['sig_temp'][foo,i])
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1)
        sigY = np.append(sigY, rass_prof['sig_temp'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*11)
        dimY = np.append(dimY, z[foo])
      
    nY = len(Y)

    # Quick check: All of the 1-sigma uncertainties from the observations
    # should have been positive. If not then abort as extra logic needs
    # to be added above...
    foo = np.where((sigY <= 0) & (Y < -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Warning: There were missing values in these obs: ' + str(feh)))
        sigY[foo] *= -1           # Presumably, the missing values had -999 for their uncertainties

    foo = np.where((sigY <= 0) & (Y > -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Error: There were negative 1-sigma uncertainties in obs: ' + str(feh) + ' Skipping sample'))
            continue

    # Compute the estimate of the forward model uncertainty (Sf).
    # This is computed at Sf = kb # Sb # transpose(Kb), where
    # B is the vector of model parameters and Sb is its covariance matrix
    # Note: right now, I will assume a perfect forward model

    Sf = np.diag(np.zeros(len(np.diag(Sy))))          # This is a matrix of zeros

    # Build the observational covariance matrix, which is a combination of
    # the uncertainties in the forward model and the observations

    Sm = Sy + Sf
    # print(Sy)
    SmInv = scipy.linalg.pinv2(Sm)

    # Get the other input variables that the forward model will need
    nX = len(z)*2                 # For both T and Q

    # Start building the first guess vector
    #    T & Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    X0 = np.copy(Xa)     # Start with the prior, and overwrite portions of it if desired
    first_guess = 'prior'
    if vip['first_guess'] == 1:
        # Use the prior as the first guess
        if verbose >= 3:
            print('Using prior as first guess')
    elif vip['first_guess'] == 2:
        # Build a first guess from the AERI-estimated surface temperture,
        # an assumed lapse rate, and a 60% RH as first guess
        if verbose >= 3:
            print('Using Tsfc with lapse rate and 60& RH as first guess')
        first_guess = 'Tsfc with lapse rate and 60% RH'
        lapserate = -7.0        # C / km
        constRH = 60.           # percent RH
        t = aeri['tsfc'][i] + z*lapserate
        p = Calcs_Conversions.inv_hypsometric(z, t+273.16, aeri['atmos_pres'][i])  # [mb]
        q = Calcs_Conversions.rh2w(t, np.ones(len(z))*constRH/100., p)
        X0 = np.concatenate([t, q, Xa[nX:nX+12]])    # T, Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    elif vip['first_guess'] == 3:
        # Get first guess from the previous retrieval, if there is one
        # If there isn't a valid prior retrieval, use prior
        if verbose >= 3:
            print('Using previous good retrieval as first guess')
        if type(xret) == list:
            for j in range(len(xret)-1, -1, -1):            # We want to use the last good retrieval, so loop from back to front
                if (xret[j]['converged'] == 1) & (aeri['hour'][i] - xret[j]['hour'] < 1.01):
                    first_guess = 'lastSample'
                    X0 = np.copy(xret[j]['Xn'])
                    break
        if first_guess == 'prior':
            if verbose >= 3:
                print('but there were no good retrievals yet')
    else:
        print('Error: Undefined first guess option')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

    # Build the first guess vector
    itern = 0
    converged = 0
    Xn = np.copy(X0)
    Fxnm1 = np.array([-999.])

    # If we are to append to the file, then I need to find the last valid
    # sample in the file, so I only process after that point...
    if ((vip['output_clobber'] == 2) & (check_clobber == 1)):
        xret, fsample, noutfilename = Output_Functions.create_xret(xret, fsample, vip, aeri, Xa, Sa, z, bands, dimY, flagY)
        check_clobber = 0
        if fsample < 0:
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        if fsample == 0:
            vip['output_clobber'] = 0
            xret = []
        if ((verbose >= 1) & (fsample > 0)):
            print(('Will append output to the file ' + noutfilename))

    # If we are in 'append' mode, then skip any AERi samples that are
    # before the last time in the xret structure. Generally, the current
    # AERI sample will always be before the last one in the xret structure,
    # except in the cases where we just started the code in append mode. If
    # that happens, then the xret structure will be (partially) populated
    # by the create_xret routine above, which gets the time from the
    # existing netCDF file. And the way we are writing the data that is all
    # that needed to be retrieved...

    if vip['output_clobber'] == 2:
        if aeri['secs'][i] <= xret[fsample-1]['secs']:
            print('  ....but was already processed (append mode)')
            continue

    cbh = aeri['cbh'][i]
    cbhflag = aeri['cbhflag'][i]

    # Define the gamma factors needed to keep the retrieval sane
    # MWR-only retrievals are more linear and thus the gfactor can be more agressive

    if vip['aeri_type'] <= -1:
        gfactor = np.array([100.,10.,3.,1.])
    else:
        if(first_guess == 'lastSample'):
            gfactor = np.array([100.,10.,3.,1.])
        else:
            gfactor = np.array([1000.,300.,100.,30.,10.,3.,1.])
    if len(gfactor) < vip['max_iterations']:
        gfactor = np.append(gfactor, np.ones(vip['max_iterations']-len(gfactor)+3))

    # Select nice round numbers to use as the wavenumber limits for
    # the LBLRTM calc, but remember that I need to pad by 50 cm-1 for FSCAN
#     lblwnum1 = np.int(np.min(wnum)-60)/100 * 100
#     lblwnum2 = (np.int(np.max(wnum)+60)/100+1)*100
    lblwnum1 = np.int((np.min(wnum)-60)/100) * 100
    lblwnum2 = (np.int((np.max(wnum)+60)/100)+1)*100
    continue_next_sample = 0          # A flag used to watch for bad jacobian calcs

    while ((itern <= vip['max_iterations']) & (converged == 0)):        # { While loop over iter

        if verbose >= 3:
            print((' Making the forward calculation for iteration ' + str(itern)))

        if os.path.exists(lbltp5):
            shutil.rmtree(lbltp5)

        if os.path.exists(lbldir):
            shutil.rmtree(lbldir)

        if os.path.exists(lbllog):
            shutil.rmtree(lbllog)

        # Update the pressure profile using the current estimate of temperature
        p = Calcs_Conversions.inv_hypsometric(z, Xn[0:int(nX/2)]+273.16, aeri['atmos_pres'][i])

        # If the trace gas profile shape is mandated to be a function of the PBL height,
        # then set that here. First, compute the current estimate of the PBL height,
        # then the coefficient and overwrite any current shape coefficient

        if itern == 0:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX/2)], p, np.sqrt(np.diag(Sa[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
        else:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX / 2)], p, np.sqrt(np.diag(Sop[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
        coef = Other_functions.get_a2_pblh(pblh)           # Get the shape coef for this PBL height
        if ((vip['retrieve_co2'] == 1) & (vip['fix_co2_shape'] == 1)):
            Xn[nX+4+2] = coef
        if ((vip['retrieve_ch4'] == 1) & (vip['fix_ch4_shape'] == 1)):
            Xn[nX+4+3+2] = coef
        if ((vip['retrieve_n2o'] == 1) & (vip['fix_n2o_shape'] == 1)):
            Xn[nX+4+6+2] = coef

        # Decorrelate the levels above the PBLH from the levels below in the prior.
        # The decorrelation factor should be between 1 and 100, I think, and read from VIP.
        # This really should only be done in the daytime though, so will need to
        # add new logic to determin local time-of-day from lat/lon position sometime.
        # prior_pblh_decorrelate_factor = 3.

        # Sa = Other_functions.covariance_matrix_decorrelate_level(prior['Sa'], z, pblh, prior_pblh_decorrelate_factor)

        # Compute its inverse of the prior
        SaInv = scipy.linalg.pinv2(Sa)

        # This function makes the forward calculation and computes the Jacobian
        # for the AERI component of the forward model
        if vip['lblrtm_jac_option'] == 1:
            if vip['aeri_type'] <= -1:
                    # If this type, then AERI data aren't being used in the retrieval
                    # so the forward calc should be missing and the Jacobian is 0
                flag = 1
                version_compute_jacobian = 'No AERI data in retrieval, so LBLRTM not used'
            else:
                print('Need to port the compute_jacobian_finitediff function. Have to abort... Sorry!!')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
                flag, Kij, FXn, wnumc, version_compute_jacobian, totaltime  = \
                           Jacobian_Functions.compute_jacobian_finitediff(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], vip['lblrtm_forward_threshold'], 
                           location['alt'], rt_extra_layers, stdatmos, 
                           verbose, debug, doapidize=True)

        elif vip['lblrtm_jac_option'] == 2:
            print('Need to port the compute_jacobian_3method function. Have to abort... Sorry!!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
            if vip['aeri_type'] <= -1:
                    # If this type, then AERI data aren't being used in the retrieval
                    # so the forward calc should be missing and the Jacobian is 0
                flag = 1
                version_compute_jacobian = 'No AERI data in retrieval, so LBLRTM not used'
            else:
                flag, Kij, FXn, wnumc, version_compute_jacobian, totaltime  = \
                           Jacobian_Functions.compute_jacobian_3method(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], vip['lblrtm_forward_threshold'], 
                           location['alt'], rt_extra_layers, stdatmos, 
                           verbose, debug, doapidize=True)

        elif vip['lblrtm_jac_option'] == 3:
            # If I am here, read in the delta OD information
            # It should be in the XaSa file. But only do it once!
            if read_deltaod <= 0:
                fid = Dataset(prior_filename, 'r')
                vid = np.where(np.array(list(fid.variables.keys())) == 'delta_od')[0]
                if len(vid) == 0:
                    fid.close()
                    print('Error: The XaSa prior file does not have the needed deltaOD info')
                    VIP_Databases_functions.abort(lbltmpdir,date)
                    sys.exit()
                awnum = fid.variables['wnum'][:]
                adeltaod = fid.variables['delta_od'][:]
                fid.close()
                foo = np.where((lblwnum1 <= awnum) & (awnum <= lblwnum2))[0]
                if len(foo) <= 0:
                    print('Error: The delta OD spectral range does not match LBLRTM spectral range')
                    VIP_Databases_functions.abort(lbltmpdir,date)
                    sys.exit()
                awnum = awnum[foo]
                adeltaod = adeltaod[foo]
                read_deltaod = 1

            if vip['aeri_type'] <= -1:
                # If this type, then AERI data aren't being used in the retrieval
                # so the forward calc should be missing and the Jacobian is 0
                wnumc = np.copy(aeri['wnum'])
                FXn = np.ones(len(wnumc))*-999.
                Kij = np.zeros((len(wnumc),len(Xn)))
                flag = 1
                version_compute_jacobian = 'No AERI data in retrieval, so LBLRTM not used'
            else:
                # Otherwise, run the forward model and compute the Jacobian
                flag, Kij, FXn, wnumc, version_compute_jacobian, totaltime  = \
                           Jacobian_Functions.compute_jacobian_deltaod(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], awnum, adeltaod, vip['lblrtm_forward_threshold'],
                           location['alt'], rt_extra_layers, stdatmos, 
                           verbose, debug, doapodize=True)

        elif vip['lblrtm_jac_option'] == 4:
                # Will use the jacobian_interpol method
            if vip['aeri_type'] <= -1:
                # If this type, then AERI data aren't being used in the retrieval
                # so the forward calc should be missing and the Jacobian is 0
                wnumc = np.copy(aeri['wnum'])
                FXn = np.ones(len(wnumc))*-999.
                Kij = np.zeros((len(wnumc),len(Xn)))
                flag = 1
                version_compute_jacobian = 'No AERI data in retrieval, so LBLRTM not used'
            else:
                # Otherwise, run the forward model and compute the Jacobian
                flag, Kij, FXn, wnumc, version_compute_jacobian, totaltime  = \
                           Jacobian_Functions.compute_jacobian_interpol(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], aeri['wnum'], vip['lblrtm_forward_threshold'],
                           location['alt'], rt_extra_layers, stdatmos, vip['lblrtm_jac_interpol_npts_wnum'], 
                           verbose, debug, doapodize=True)
        else:
            print('Error: Undefined jacobian option selected')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

        # If the Jacobian did not compute properly (i.e., an error occurred)
        # then we need to abort
        if flag == 0:
            print(' -- Skipping this sample due to issue with LBLRTM Jacobian (likely bad input profile)')
            continue_next_sample = 1
            break

        # Select the wavenumber indices to use
        w1idx, junk = Other_functions.find_wnum_idx(wnumc, bands)
        if len(w1idx) != len(w0idx):
            print('Problem with wnum indices1')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        wnumc = wnumc[w1idx]
        FXn = FXn[w1idx]
        Kij = Kij[w1idx,:]

        # Are there missing values from the AERI? If so,then we want to make the
        # forward model calculation have the same value and put no sensitivity
        # in the Jacobian there so that the retrieval is unaffected (this is really for
        # aeri_type = -1)

        foo = np.where(flagY == 1)[0]
        bar = np.where(Y[foo] < -900)[0]
        if len(bar) > 0:
            FXn[bar] = np.copy(Y[foo[bar]])
            for gg in range(len(bar)):
                Kij[bar[gg],:] = 0.

        # Now start processing the other observation types that might be in the obs vector

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-zenith portion of the observation vector
        foo = np.where(flagY == 2)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1.0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                        # Eventually, I need to add location.alt/1000. to the z(gg) and
                        # the rt_extra_layers(gg) in the next two lines.  But I need to resolve
                        # the issue with the surface offset in monortm_v5 first. See comment
                        # in the routine compute_jacobian_microwavescan_3method.
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0



            if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(lbltmpdir + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwr['freq'])))
                for gg in range(len(mwr['freq'])):
                    lun.write('{:7.3f}\n'.format(mwr['freq'][gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0

                # Run the forward model and compute the Jacobian
            if vip['monortm_jac_option'] == 1:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_finitediff(Xn, p, z,
                            mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                            fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            elif vip['monortm_jac_option'] == 2:
                    flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_3method(Xn, p, z,
                            mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                            fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            else:
                print('Error: Undefined option for monortm_jac_option')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # If the Jacobian did not compute properly (i.e., an error ocurred),
            # then we need to abort
            if flag == 0:
                print('-- Skipping this sample due to issue with MonoRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break

            # Check to see if I am replicating the observation; if so, then adjust the
            # forward calc and Jacobian appropriately (they are the same value repeated)
            # TODO - There's an issue here when the TB obs are replicated
            tmpFF = np.copy(FF)
            tmpKK = np.copy(KK)
            for gg in range(1,vip['mwr_tb_replicate']):
                tmpFF = np.append(tmpFF, FF)
                tmpKK = np.append(tmpKK, KK, axis = 0)
            FF = np.copy(tmpFF)
            KK = np.copy(tmpKK)

            # Now the size of the forward calculation should be the correct size to match
            # the number of MWR observations in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external temperature profiler portion of the observation vector
        foo = np.where(flagY == 3)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_temp_profiler(Xn, p, z,
                            ext_prof['tempminht'], ext_prof['tempmaxht'], ext_prof['temp_type'])

            if flag == 0:
                print('Problem computing the Jacobian for the external temp profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external temp profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the external profilers? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external water vapor profiler portion of the observation vector
        foo = np.where(flagY == 4)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_wv_profiler(Xn, p, z,
                            ext_prof['wvminht'], ext_prof['wvmaxht'], ext_prof['wv_type'], ext_prof['wvmultiplier'])


            if flag == 0:
                print('Problem computing the Jacobian for the external wv profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external wv profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the external profilers? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external water vapor profiler portion ofthe observation vector
        foo = np.where((flagY == 5) | (flagY == 6))[0]
        if len(foo) > 0:
            units = ''
            barT = np.where(flagY == 5)[0]
            barQ = np.where(flagY == 6)[0]

            if ((len(barT) > 0) & (len(barQ) > 0)):
                units = np.append(np.array([ext_tseries['tunit']]*len(barT)), np.array([ext_tseries['qunit']]*len(barQ)))
            elif ((len(barT) > 0) & (len(barQ) == 0)):
                units = np.array([ext_tseries['tunit']]*len(barT))
            elif ((len(barT) == 0) & (len(barQ) > 0)):
                units = np.array([ext_tseries['qunit']]*len(barQ))
            else:
                print('This absolutely should not happen, as I should be finding some units')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_sfc_met(Xn, p, z,
                             ext_tseries['sfc_relative_height'], units, vip['prior_chimney_ht'])

            if flag == 0:
                print('Problem computing the Jacobian for the external surface met data. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external surface met')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values in the surface met data? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # NWP model temperature profiler portion of the observation vector
        foo = np.where(flagY == 7)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_temp_profiler(Xn, p, z,
                            mod_prof['tempminht'], mod_prof['tempmaxht'], mod_prof['temp_type'])

            if flag == 0:
                print('Problem computing the Jacobian for the NWP temp profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the NWP temp profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the NWP profile? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # NWP model water vapor profiler portion of the observation vector
        foo = np.where(flagY == 8)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_wv_profiler(Xn, p, z,
                            mod_prof['wvminht'], mod_prof['wvmaxht'], mod_prof['wv_type'], mod_prof['wvmultiplier'])

            if flag == 0:
                print('Problem computing the Jacobian for the NWP wv profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the NWP wv profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the NWP profile? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # in-situ surface co2 measurements of the observation vector
        foo = np.where(flagY == 9)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_sfc_co2(Xn, p, z,
                        ext_tseries['co2_sfc_relative_height'], vip['retrieve_co2'],
                        vip['fix_co2_shape'], ext_tseries['nptsCO2'])

            if flag == 0:
                print('Problem computing the Jacobian for the in-situ surface co2 data. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the in-situ surfac co2 data.')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the in-situ co2 data? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-scan portion of the observation vector
        foo = np.where(flagY == 10)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Note this is the same config file used in the MWR-zenith, so it
                # may have already been created. (The config is the same in both
                # -zenith and -scan)

                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1,0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                        # See the comment above about "compute_jacobian_microwavescan_3method"
                        # associated with the development of the monortm_config file. 
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0

            if create_monortm_sfreq == 1:
                # Create the MonoRTM frequency file
                lun = open(lbltmpdir + '/' + monortm_sfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwrscan['freq'])))
                for gg in range(len(mwrscan['freq'])):
                    lun.write('{:7.3f}\n'.format(mwrscan['freq'][gg]))
                lun.close()

                # Turn the flag off, as we only need to create thee files once
                create_monortm_sfreq = 0

            # Run the forward model and compute the Jacobian
            if vip['monortm_jac_option'] == 1:
                print('Error: DDT has not enabled finite differencing in the MWR-scan jacobian yet')
            elif vip['monortm_jac_option'] == 2:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwavescan_3method(Xn, p, z,
                                        mwrscan, cbh, vip, lbltmpdir, monortm_tfile, monortm_sexec,
                                        fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], 
                                        verbose)
            else:
                print('Error: Undefined option for monortm_jac_option')

            # If the Jacobian did not compute properly (i.e., an error occurred)
            # then we need to abort

            if flag == 0:
                print(' -- Skipping this sample due to issure with MonoRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break

            # Now the size fo the forward calculation should be the correct size to match the
            # the number of MWR-scan observation in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer scan')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-scan portion of the observation vector
        foo = np.where(flagY == 11)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_temp_profiler(Xn, p, z,
                            rass_prof['tempminht'], rass_prof['tempmaxht'], rass_prof['temp_type'])

            if flag == 0:
                print('Problem computing the Jacobian for the RASS profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the RASS profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the RASS? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)


        ########
        # Done computing forward calculation and Jacobians. Now the retrieval math
        ########

        # Compute the L-curve values to determine an appropriate point for gamma
        use_L_curve = 0
        if use_L_curve > 0:
            # Starting here, I am following the Carissimo et al. logic hete
            if (itern == 0) & verbose >= 2:
                print('Using the L-curve method to optimize gamma')
                ggamma = np.arange(101)*0.10 - 5     # Values from -5 to +5
                ggamma = 10.**ggamma        # this is the range of values I want: 10^(-5) to 10^(+5)

                gfac = Other_functions.lcurve(ggamma, flagY, Y, FXn, Kij, Xn, Xa, Sa, Sm, z)
        else:
            gfac = gfactor[itern]

        # Retrieval Calculations
        B      = (gfac * SaInv) + Kij.T.dot(SmInv).dot(Kij)
        Binv   = scipy.linalg.pinv2(B)
        Gain   = Binv.dot(Kij.T).dot(SmInv)
        Xnp1   = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        Sop    = Binv.dot(gfac*gfac*SaInv + Kij.T.dot(SmInv).dot(Kij)).dot(Binv)
        SopInv = scipy.linalg.pinv2(Sop)
        Akern  = (Binv.dot(Kij.T).dot(SmInv).dot(Kij)).T

        if(vip['max_iterations'] == 0):
            if(vip['aeri_type'] > 0):
                print(f'        DDT - compute_jacobian_xx took {totaltime:.1f} seconds')
            print('Special debug mode -- writing variables from retrieval calcs in the directory '+vip['lbl_temp_dir'])
            #Output_Functions.write_variable(    B,vip['lbl_temp_dir']+'/tropoe_python_output.B.cdf')
            #Output_Functions.write_variable(SaInv,vip['lbl_temp_dir']+'/tropoe_python_output.SaInv.cdf')
            #Output_Functions.write_variable( gfac,vip['lbl_temp_dir']+'/tropoe_python_output.gfac.cdf')
            #Output_Functions.write_variable(   Xa,vip['lbl_temp_dir']+'/tropoe_python_output.Xa.cdf')
            #Output_Functions.write_variable( Gain,vip['lbl_temp_dir']+'/tropoe_python_output.Gain.cdf')
            #Output_Functions.write_variable(  Sop,vip['lbl_temp_dir']+'/tropoe_python_output.Sop.cdf')
            #Output_Functions.write_variable( Xnp1,vip['lbl_temp_dir']+'/tropoe_python_output.Xnp1.cdf')
            #Output_Functions.write_variable(Akern,vip['lbl_temp_dir']+'/tropoe_python_output.Akern.cdf')
            Output_Functions.write_variable(SmInv,vip['lbl_temp_dir']+'/tropoe_python_output.SmInv.cdf')
            Output_Functions.write_variable(  Kij,vip['lbl_temp_dir']+'/tropoe_python_output.Kij.cdf')
            Output_Functions.write_variable(    Y,vip['lbl_temp_dir']+'/tropoe_python_output.Y.cdf')
            Output_Functions.write_variable(  FXn,vip['lbl_temp_dir']+'/tropoe_python_output.FXn.cdf')
            #sys.exit()

        # If we are trying to fix the shape of the TG profiles as a function of the
        # PBLH, then we need to make a special tweak here. The gain matrix for the
        # factor(s) will be zero, which would make the next iteration have the shape
        # factor in the prior. But I don't want to be changing the prior with each iteration,
        # as that will impact the "append" option (if we are using that). So we need this
        # stub of code to do the same thing

        if ((vip['retrieve_co2'] == 1) & (vip['fix_co2_shape'] == 1)):
            Xnp1[nX+4+2] = np.copy(Xn[nX+4+2])
        if ((vip['retrieve_ch4'] == 1) & (vip['fix_ch4_shape'] == 1)):
            Xnp1[nX+4+5] = np.copy(Xn[nX+4+5])
        if ((vip['retrieve_n2o'] == 1) & (vip['fix_n2o_shape'] == 1)):
            Xnp1[nX+4+8] = np.copy(Xn[nX+4+8])

        # Look for NaN values in this updated state vector. They should not
        # exist, but if they do, then let's stop the code here to allow
        # me to look at it. Not optimal solution for operation code
        # though, as it really should output a flagged result or abort/
        foo = np.where(np.isnan(Xnp1))[0]          # DDT
        if len(foo) > 0:
            print('Stopping for NaN issue 1')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

        # Compute some information content numbers. The DFS will be computed
        # as the [total, temp, WVMR, LWP, ReffL, TauI, ReffI, co2, ch4, n2o]
        tmp = np.diag(Akern)
        dfs = np.array([np.sum(tmp), np.sum(tmp[0:int(nX/2)]), np.sum(tmp[int(nX/2):nX]), tmp[nX],
                    tmp[nX+1], tmp[nX+2], tmp[nX+3], tmp[nX+4], tmp[nX+5], tmp[nX+6],
                    tmp[nX+7], tmp[nX+8], tmp[nX+9], tmp[nX+10], tmp[nX+11], tmp[nX+12]])

        sic = 0.5 * np.log(scipy.linalg.det(Sa.dot(SopInv)))

        vres,cdfs = Other_functions.compute_vres_from_akern(Akern, z, do_cdfs=True)
        # Compute the N-form and M-form convergence criteria (X and Y spaces, resp)
        if itern == 0:
        # Set the initial RMS and di2 values to large numbers

            old_rmsa = 1e20          # RMS for all observations
            old_rmsr = 1e20          # RMS for only the AERI and MWR radiance obs
            old_di2m = 1e20          # di-squared number

        di2n = ((Xn[:,None]-Xnp1).T.dot(SopInv).dot(Xn[:,None]-Xnp1))[0,0]
        if len(Fxnm1) == nY:
            di2m = ((FXn[:,None] - Fxnm1[:,None]).T.dot(
                scipy.linalg.pinv2(Kij.dot(Sop).dot(Kij.T)+Sm)).dot(
                FXn[:,None] - Fxnm1[:,None]))[0,0]
        else:
            di2m = 9.0e9

        # Perform the RH_limit test (i.e., make sure thew WVMR Is not too large
        # such that RH > 100%)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for RH > 100%')
        rh = Calcs_Conversions.w2rh(np.squeeze(Xnp1[int(nX/2):nX]), p, np.squeeze(Xnp1[0:int(nX/2)]),0) * 100   # units are %RH
        feh = np.where(rh > 100)[0]
        if len(feh) > 0:
            if verbose >= 3:
                print('RH is above 100% somewhere in this profile -- setting it to 100%')
            rh[feh] = 100.
            Xnp1[int(nX/2):nX,0] = Calcs_Conversions.rh2w(np.squeeze(Xnp1[0:int(nX/2)]), rh/100., p)

        # Perform the monotonically ascending potential temperature test (i.e
        # make sure that theta never decreases with height)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for decreasing theata with height')

        # Multiply WVMR by zero to get theta, not theta-v
        theta = Calcs_Conversions.t2theta(np.squeeze(Xnp1[0:int(nX/2)]), 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # This creates the maximum theta
        for ii in range(len(theta))[1:]:
            if ((theta[ii] <= theta[ii-1]) & (z[ii] > vip['superadiabatic_maxht'])):
                theta[ii] = theta[ii-1]

        # Multiply WVMR by zero to work with theta, not theta-v
        Xnp1[0:int(nX/2),0] = Calcs_Conversions.theta2t(theta, 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # Make sure we don't get any nonphysical values here that would
        # make the next iteration of the LBLRTM croak
        multiplier = 5.

        feh = np.arange(int(nX/2)) + int(nX/2)
        foo = np.where((Xnp1[feh,0] < minQ) | (Xnp1[feh,0] < Xa[feh] - multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]

        if len(foo) > 0:
            Xnp1[feh[foo],0] = np.nanmax([Xa[feh[foo]] - np.sqrt(np.diag(Sa)[feh[foo]]), np.ones(len(Xa[feh[foo]]))*minQ], axis = 0)

        foo = np.where((Xnp1[feh,0] > maxQ) | (Xnp1[feh,0] > Xa[feh] + multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]

        if len(foo) > 0:
           Xnp1[feh[foo],0] = np.nanmin([Xa[feh[foo]] + np.sqrt(np.diag(Sa)[feh[foo]]), np.ones(len(Xa[feh[foo]]))*maxQ], axis = 0)

        if dolcloud == 1:
            Xnp1[nX,0] = np.nanmax([Xnp1[nX],0])
            Xnp1[nX+1,0] = np.nanmax([Xnp1[nX+1], vip['prior_lReff_mn']-multiplier*vip['prior_lReff_sd'], minLReff])
            Xnp1[nX+1,0] = np.nanmin([Xnp1[nX+1], vip['prior_lReff_mn']+multiplier*vip['prior_lReff_sd'], maxLReff-2])

        if doicloud == 1:
            Xnp1[nX+2,0] = np.nanmax([Xnp1[nX+2],0])
            Xnp1[nX+3,0] = np.nanmax([Xnp1[nX+3], vip['prior_iReff_mn']-multiplier*vip['prior_iReff_sd'], miniReff])
            Xnp1[nX+3,0] = np.nanmin([Xnp1[nX+3], vip['prior_iReff_mn']+multiplier*vip['prior_iReff_sd'], maxiReff-2])

        if doco2 > 0:
            Xnp1[nX+4,0] = np.nanmax([Xnp1[nX+4], vip['prior_co2_mn'][0]-multiplier*vip['prior_co2_sd'][0], 0])
            Xnp1[nX+4,0] = np.nanmin([Xnp1[nX+4], vip['prior_co2_mn'][0]+multiplier*vip['prior_co2_sd'][0]])

            Xnp1[nX+5,0] = np.nanmax([Xnp1[nX+5], vip['prior_co2_mn'][1]-multiplier*vip['prior_co2_sd'][1]])
            Xnp1[nX+5,0] = np.nanmin([Xnp1[nX+5], vip['prior_co2_mn'][1]+multiplier*vip['prior_co2_sd'][1]])

            if Xnp1[nX+4] + Xnp1[nX+5]  < 0:
                Xnp1[nX+5] = -Xnp1[nX+4]

            if doco2 == 1:
                Xnp1[nX+6,0] = np.nanmax([Xnp1[nX+6], vip['prior_co2_mn'][2] - multiplier*vip['prior_co2_sd'][2], -20])
                Xnp1[nX+6,0] = np.nanmax([Xnp1[nX+6], vip['prior_co2_mn'][2] + multiplier*vip['prior_co2_sd'][2], -1])
            else:
                if Xnp1[nX+6] < vip['min_PBL_height']:
                    Xnp1[nX+6,0] = vip['min_PBL_height']
                if Xnp1[nX+6] > vip['max_PBL_height']:
                    Xnp1[nX+6,0] = vip['max_PBL_height']

        if doch4 > 0:
            Xnp1[nX+7,0] = np.nanmax([Xnp1[nX+7], vip['prior_ch4_mn'][0] - multiplier*vip['prior_ch4_sd'][0], 0])
            Xnp1[nX+7,0] = np.nanmin([Xnp1[nX+7], vip['prior_ch4_mn'][0] + multiplier*vip['prior_ch4_sd'][0]])

            Xnp1[nX+8,0] = np.nanmax([Xnp1[nX+8], vip['prior_ch4_mn'][1]-multiplier*vip['prior_ch4_sd'][1]])
            Xnp1[nX+8,0] = np.nanmin([Xnp1[nX+8], vip['prior_ch4_mn'][1]+multiplier*vip['prior_ch4_sd'][1]])

            if Xnp1[nX+7] + Xnp1[nX+8]  < 0:
                Xnp1[nX+8,0] = -Xnp1[nX+7]

            if doch4 == 1:
                Xnp1[nX+9,0] = np.nanmax([Xnp1[nX+9], vip['prior_ch4_mn'][2] - multiplier*vip['prior_ch4_sd'][2], -20])
                Xnp1[nX+9,0] = np.nanmax([Xnp1[nX+9], vip['prior_ch4_mn'][2] + multiplier*vip['prior_ch4_sd'][2], -1])

            else:
                if Xnp1[nX+9] < vip['min_PBL_height']:
                    Xnp1[nX+9,0] = vip['min_PBL_height']
                if Xnp1[nX+9] > vip['max_PBL_height']:
                    Xnp1[nX+9,0] = vip['max_PBL_height']

        if don2o > 0:
            Xnp1[nX+10,0] = np.nanmax([Xnp1[nX+10], vip['prior_n2o_mn'][0] - multiplier*vip['prior_n2o_sd'][0], 0])
            Xnp1[nX+10,0] = np.nanmin([Xnp1[nX+10], vip['prior_n2o_mn'][0] + multiplier*vip['prior_n2o_sd'][0]])

            Xnp1[nX+11,0] = np.nanmax([Xnp1[nX+11], vip['prior_n2o_mn'][1]-multiplier*vip['prior_n2o_sd'][1]])
            Xnp1[nX+11,0] = np.nanmin([Xnp1[nX+11], vip['prior_n2o_mn'][1]+multiplier*vip['prior_n2o_sd'][1]])

            if Xnp1[nX+10] + Xnp1[nX+11]  < 0:
                Xnp1[nX+11,0] = -Xnp1[nX+10]

            if don2o == 1:
                Xnp1[nX+12,0] = np.nanmax([Xnp1[nX+12], vip['prior_n2o_mn'][2] - multiplier*vip['prior_n2o_sd'][2], -20])
                Xnp1[nX+12,0] = np.nanmax([Xnp1[nX+12], vip['prior_n2o_mn'][2] + multiplier*vip['prior_n2o_sd'][2], -1])

            else:
                if Xnp1[nX+12] < vip['min_PBL_height']:
                    Xnp1[nX+12] = vip['min_PBL_height']
                if Xnp1[nX+12,0] > vip['max_PBL_height']:
                    Xnp1[nX+12,0] = vip['max_PBL_height']

        # Compute the RMS difference between the observation and the
        # forward calculation. However, this will be the relative RMS
        # difference (normalizing by the observation error here), because I
        # am mixing units from all of the different types of observation
        # But I will also compute the chi-square value of the obs vs. F(Xn)

        chi2 = np.sqrt(np.sum(((Y - FXn)/ Y)**2) / np.float(nY))
        rmsa = np.sqrt(np.sum(((Y - FXn)/sigY)**2) / np.float(nY))
        feh = np.where((flagY == 1) | (flagY == 2) & (Y > -900))[0]
        if len(feh) > 0:
            rmsr = np.sqrt(np.sum(((Y[feh] - FXn[feh])/sigY[feh])**2) / np.float(len(feh)))
        else:
            rmsr = -999.

        # I decided to just change the metric to look at the normalized
        # distance to the climatological prior, but I will let it have either
        # positive or negative values. ONly compute this for the Tq part though

        feh = np.arange(nX)
        rmsp = np.mean( (Xa[feh] - Xn[feh])/sig_Xa[feh] )

        ########################
        # Add doplot code here #
        ########################

        if doplot:
            if itern == 0:
                xx = np.array([X0.copy()])
                fxx = np.array([Y.copy()])
                frms = np.array([999.])


            xx = np.append(xx, np.array([Xn.copy()]), axis=0)
            fxx = np.append(fxx, np.array([FXn.copy()]), axis=0)
            frms = np.append(frms, rmsp)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            ylim=(0, 3)
            # Temperature plot
            ax1.plot(xx[0, 0:int(nX/2)], z, label='prior')
            for ii in range(1, xx.shape[0]):
                ax1.scatter(xx[ii, 0:int(nX/2)], z, color=f"C{ii}", marker='x', label=f'iter{ii}')
            ax1.plot(Xnp1[0:int(nX/2)], z, linewidth=2, color='k')
            ax1.set_ylim(ylim)
            ax1.set_xlim(-30, 40)
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Altitude (km AGL)")
            ax1.legend()

            ax2.plot(xx[0, int(nX/2):nX], z)
            for ii in range(1, xx.shape[0]):
                ax2.scatter(xx[ii, int(nX/2):nX], z, color=f"C{ii}", marker='x')
            ax2.plot(Xnp1[int(nX/2):nX], z, linewidth=2, color='k')
            ax2.set_ylim(ylim)
            ax2.set_xlim(0, maxQ)
            ax2.set_xlabel("WVMR")
            ax2.set_ylabel("Altitude (km AGL)")

            ax3.hlines(0, 0, len(Y), color='k')
            for ii in range(1, xx.shape[0]):
                ax3.plot(Y-fxx[ii], color=f"C{ii}")
            ax3.set_ylim(-10, 10)
            ax3.set_xlabel('Radiance Index')
            ax3.set_ylabel('Radiance Diff (RU)')

            plt.tight_layout()
            #ax1.plot(xx[])
            plt.savefig(f'temp_output_{itern}.png')
            plt.close()
        # Capture the iteration with the best RMS value
        if rmsa <= old_rmsa:
            old_rmsa = rmsa
            old_rmsr = rmsr
            old_iter = itern

        # Check for NaNs in the next iteration. If they exist, then
        # use the last valid sample as the solution and exit.
        foo = np.where(np.isnan(Xnp1))[0]
        if len(foo) > 0:
            print('Warning: Found NaNs in the next iteration -- using last iter')
            if itern == 0:
                print('Wow -- I never thought this could happen')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            converged = 9                             # Converged in "bad NaN sense
            Xn = np.copy(xsamp[itern-1]['Xn'])
            FXn = np.copy(xsamp[itern-1]['FXn'])
            Sop = np.copy(xsamp[itern-1]['Sop'])
            K = np.copy(xsamp[itern-1]['K'])
            Gain = np.copy(xsamp[itern-1]['Gain'])
            Akern = np.copy(xsamp[itern-1]['Akern'])
            vres = np.copy(xsamp[itern-1]['vres'])
            gfac = xsamp[itern-1]['gamma']
            sic = xsamp[itern-1]['sic']
            dfs = np.copy(xsamp[itern-1]['dfs'])
            cdfs = np.copy(xsamp[itern-1]['cdfs'])
            di2m = xsamp[itern-1]['di2m']
            rmsa = xsamp[itern-1]['rmsa']
            rmsr = xsamp[itern-1]['rmsr']
            rmsp = xsamp[itern-1]['rmsp']
            chi2 = xsamp[itern-1]['chi2']
            itern = -1

        elif itern > 1:
            # Test for "convergence by looking at the best RMS value
            if ((rmsa > np.sqrt(gfactor[old_iter])*old_rmsa) & (old_iter > 0)):
                converged = 2                   # Converged in "rms increased drastically" sense

                Xn = np.copy(xsamp[old_iter]['Xn'])
                FXn = np.copy(xsamp[old_iter]['FXn'])
                Sop = np.copy(xsamp[old_iter]['Sop'])
                K = np.copy(xsamp[old_iter]['K'])
                Gain = np.copy(xsamp[old_iter]['Gain'])
                Akern = np.copy(xsamp[old_iter]['Akern'])
                vres = np.copy(xsamp[old_iter]['vres'])
                gfac = xsamp[old_iter]['gamma']
                sic = xsamp[old_iter]['sic']
                dfs = np.copy(xsamp[old_iter]['dfs'])
                cdfs = np.copy(xsamp[old_iter]['cdfs'])
                di2m = xsamp[old_iter]['di2m']
                rmsa = xsamp[old_iter]['rmsa']
                rmsr = xsamp[old_iter]['rmsr']
                rmsp = xsamp[old_iter]['rmsp']
                chi2 = xsamp[old_iter]['chi2']
                itern = old_iter

            # But also check for convergence in the normal manner
            if ((gfactor[itern-1] <= 1) & (gfactor[itern] == 1)):
                if di2m < cvgmult * nY:
                    converged = 1                    # Converged in "classical sense"

        prev_di2m = di2m

        if converged == 0:
            Xn = np.copy(Xnp1[:,0])
            Fxnm1 = np.copy(FXn)
            itern += 1
            if verbose >= 1:
                print(f"    iter is {itern:2d}, di2m is {di2m:.3e}, and RMS is {rmsa:.3e}")

        # Place the data into a structure
        xtmp = {'idx':i, 'secs':aeri['secs'][i], 'ymd':aeri['ymd'][i], 'hour':aeri['hour'][i],
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':aeri['hatchopen'][i],
                'cbh':cbh, 'cbhflag':cbhflag,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2m':di2m, 'rmsa':rmsa,
                'rmsr':rmsr, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged}

        # And store each iteration in case I would like to investigate how
        # the retrieval functioned in a sample-by-sample way

        if itern == 1:
            xsamp = [copy.deepcopy(xtmp)]
        else:
            xsamp.append(copy.deepcopy(xtmp))

    if continue_next_sample == 1:
        continue         # This was set if the Jacobian could not be computed

    # If the retrieval converged, then let's store the various
    # pieces of information for later. Otherwise, let's just move on...
    if converged == 1:
        print('    Converged! (di2m << nY)')
    elif converged == 2:
        print('    Converged (best RMS as RMS drastically increased)')
    elif converged == 9:
        print('    Converged (found NaN in Xnp1 so abort sample)')
    else:

        # If the retrieval did not converged but performed max_iter iterations
        # means that the RMS didn't really increase drastically at any one step.
        # Let's select the sample that has the best RMS but weight the value
        # so that we are picking it towards the end ofthe iterations (use gamma
        # to do so), and save it

        vval = []
        for samp in xsamp:
            vval.append(np.sqrt(samp['gamma']) * samp['rmsa'])
        vval = np.array(vval)
        foo = np.where(vval < np.min(vval)*1.00001)[0]
        converged = 3           # Converged in "best rms after max_iter" sense
        itern = int(foo[0])
        Xn = np.copy(xsamp[itern]['Xn'])
        FXn = np.copy(xsamp[itern]['FXn'])
        Sop = np.copy(xsamp[itern]['Sop'])
        K = np.copy(xsamp[itern]['K'])
        Gain = np.copy(xsamp[itern]['Gain'])
        Akern = np.copy(xsamp[itern]['Akern'])
        vres = np.copy(xsamp[itern]['vres'])
        gfac = xsamp[itern]['gamma']
        sic = xsamp[itern]['sic']
        dfs = np.copy(xsamp[itern]['dfs'])
        cdfs = np.copy(xsamp[itern]['cdfs'])
        di2m = xsamp[itern]['di2m']
        rmsa = xsamp[itern]['rmsa']
        rmsr = xsamp[itern]['rmsr']
        rmsp = xsamp[itern]['rmsp']
        chi2 = xsamp[itern]['chi2']
        xtmp = {'idx':i, 'secs':aeri['secs'][i], 'ymd':aeri['ymd'][i], 'hour':aeri['hour'][i],
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':aeri['hatchopen'][i],
                'cbh':cbh, 'cbhflag':cbhflag,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2m':di2m, 'rmsa':rmsa,
                'rmsr':rmsr, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged}

        xsamp.append(xtmp)
        print('Converged! (best RMS after max_iter)')

    # Store the data, regardless whether it converges or not
    if xret == []:
        xret = [copy.deepcopy(xsamp[len(xsamp)-1])]
    else:
        xret.append(copy.deepcopy(xsamp[len(xsamp)-1]))

    # Note that I will also save all of the iterations for the
    # last time sample that was processed (i.e., "xsamp")
    if version == '':
        try:
            version = globatt['algorithm_version'] + version_compute_jacobian
        except:
            print('Did not find globatt version')
            version = 'Unknown Version' + version_compute_jacobian

    endtime = datetime.now()

    # Now store all of the data out
    ################################
    # IDL SAVE FILE CODE GOES HERE #
    ################################
    if debug == 1:
        import pickle
        # Create the .IDL file only if DEBUG is set
        # (Main reason is to get the xsamp data that show each iteration)
        dt = datetime.utcfromtimestamp(aeri['secs'][i])
        hr = int(shour)*100+int(((shour-int(shour))*60+.5))
        savename = dt.strftime(f"{vip['output_path']}/{vip['output_rootname']}.%Y%m%d.{hr}.%H%M.pkl")
        out = {'xret': xret, 'xsamp': xsamp, 'bands': bands, 'vip':vip,
               'doco2':doco2, 'doch4': doch4, 'don2o':don2o,
               'dolcloud':dolcloud,'Sa':Sa, 'Xa':Xa, 'nsonde_prior':nsonde_prior,
               'comment_prior':comment_prior, 'aeri':aeri, 'shour':shour,'ehour':ehour,
               'starttime':starttime, 'endtime':endtime, 'version':__version__}
        with open(savename, 'wb') as fh:
            pickle.dump(out, fh)


    # Determine the QC of the sample
    # First look for a hatch that is isn't fully open
    if ((0.8 > xret[fsample]['hatchopen']) & (xret[fsample]['hatchopen'] > 1.2)):
        xret[fsample]['qcflag'] = 1
    # Then look for a retrieval that didn't converge
    if xret[fsample]['converged'] != 1:
        xret[fsample]['qcflag'] = 2
    # Then look for a retrieval where the RMS is too large
    if xret[fsample]['rmsa'] > vip['qc_rms_value']:
        xret[fsample]['qcflag'] = 3

    # Compute the various convective indices and other useful data
    # Write the data into the netCDF file

    success, noutfilename = Output_Functions.write_output(vip, ext_prof, mod_prof, rass_prof, ext_tseries,
              globatt, xret, prior, fsample, version, (endtime-starttime).total_seconds(),
              modeflag, noutfilename, location, cbh_string, verbose)

    if success == 0:
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

    already_saved = 1
    fsample += 1




shutil.rmtree(lbltmp)

totaltime = (endtime - starttime).total_seconds()

print(('Processing took ' + str(totaltime) + ' seconds'))

shutil.rmtree(lbltmpdir)

# Successful exit
print(('>>> TROPoe retrieval on ' + str(date) + ' ended properly <<<'))
print('--------------------------------------------------------------------')
print(' ')
