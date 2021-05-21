import numpy as np
import glob
from netCDF4 import Dataset
import calendar
from datetime import datetime, timedelta

import Other_functions
import VIP_Databases_functions
import Calcs_Conversions
import Jacobian_Functions
import Output_Functions


################################################################################
# This file containts the following functions:
# read_all_data()
# read_mwr()
# read_mwrscan()
# read_aeri_eng()
# read_aeri_ch()
# read_aeri_sum()
# read_vceil()
# grid_aeri()
# grid_mwr()
# grid_mwrscan()
# read_external_profile_data()'
# read_external_timeseries()
################################################################################

################################################################################
# This function reads the AERI channel data file
################################################################################

def read_aeri_ch(path,date,aeri_type,fv,fa,engsecs,engtemp,bbcavfactor,
                 get_aeri_missingDataFlag, verbose):
                 
    if verbose >= 2:
        print('Reading aeri_ch data in ' + path)
    err = {'success':0}
    if aeri_type <= 0:
        print(('This piece of code should not be exercised, as this option should be screened ' +
               'in read_aeri_eng() earlier'))
        return err
    elif ((aeri_type == 1) | (aeri_type == 4)):
        filename = path + '/' + '*aeri*ch*' + str(date) + '*cdf'
    elif aeri_type == 2:
        filename = path + '/' + '*' + str(date % 2000000) + 'C1_rnc.cdf'
    elif aeri_type == 3:
        filename = path + '/' + '*' + str(date % 2000000) + 'C1.RNC.cdf'
    else:
        print('Error in read_aeri_ch: unable to decode aeri_type')
        return err
    
    if verbose >= 3:
        print('Looking for AERI channel data as ' + filename)
    
    files = glob.glob(filename)
    if len(files) == 0:
        print('Error: Unable to find any AERI channel data -- aborting')
        return err
    
    for jj in range(len(files)):
        fid = Dataset(files[jj])
        bt = fid.variables['base_time'][:]
        to = fid.variables['time_offset'][:]
        vid = np.where(np.array(list(fid.variables.keys())) == 'wnum')[0]
        if len(vid) > 0:
            wnum = fid.variables['wnum'][:]
        else:
            vid = np.where(np.array(list(fid.variables.keys())) == 'wnum1')[0]
            if len(vid) > 0:
                wnum = fid.variables['wnum1'][:]
            else:
                print('Error in read_aeri_ch: unable to find either "wnum" or "wnum1 -- aborting')
                return err
        
        xmrad = fid.variables['mean_rad'][:]
        if len(np.where(np.array(list(fid.variables.keys())) == 'hatchOpen')[0]) > 0:
            xhatchOpen = fid.variables['hatchOpen'][:]
        elif  len(np.where(np.array(list(fid.variables.keys())) == 'hatchIndicator')[0]) > 0:
            #The AERI-01 has a hatchIndicator flag instead of hatchOpen,
            #but I need to change some of the values so that they are consistent
            
            xhatchOpen = fid.variables['hatchIndicator'][:]
            foo = np.where((xhatchOpen >= 1.5) & (xhatchOpen < 2.5))[0]
            if len(foo) > 0:
                xhatchOpen[foo] = 0   
            foo = np.where((xhatchOpen > 2.5) | (xhatchOpen < -0.5))[0]
            if len(foo) > 0:
                xhatchOpen[foo] = 3
        else:
            print(('Warning: Unable to find AERI hatchOpen or hatchIndicator field in ' +
                  'data file -- assuming hatch is always open'))
            xhatchOpen = np.ones(len(to))
        
        xbbsupport = fid.variables['BBsupportStructureTemp'][:]
        xcalibambt = fid.variables['calibrationAmbientTemp'][:]
        xcalibcbbt = fid.variables['calibrationCBBtemp'][:]
        xcalibhbbt = fid.variables['calibrationHBBtemp'][:]
        xambPres = fid.variables['atmosphericPressure'][:]
        
        #Read in the field "missingDataFlag". If it does not exist, then abort
        if get_aeri_missingDataFlag == 1:
            if len(np.where(np.array(list(fid.variables.keys())) == 'missingDataFlag')[0]) > 0:
                xmissingDataFlag = fid.variables['missingDataFlag'][:]
                xmissingDataFlag = xmissingDataFlag.astype('int')
            else:
                print('Error in read_aeri_ch: unable to find the field missingDataFlag')
                return err
        else:
            xmissingDataFlag = np.zeros(len(to))
        fid.close()
        
        #Append the data into single arrays
        if jj == 0:
            secs = bt+to
            mrad = np.copy(xmrad)
            hatchOpen = np.copy(xhatchOpen)
            bbsupport = np.copy(xbbsupport)
            calibambt = np.copy(xcalibambt)
            calibcbbt = np.copy(xcalibcbbt)
            calibhbbt = np.copy(xcalibhbbt)
            ambPres = np.copy(xambPres)
            missingDataFlag = np.copy(xmissingDataFlag)
        else:
            secs = np.append(secs,bt+to)
            mrad = np.append(mrad,xmrad, axis = 0)
            hatchOpen = np.append(hatchOpen,xhatchOpen)
            bbsupport = np.append(bbsupport,xbbsupport)
            calibambt = np.append(calibambt,xcalibambt)
            calibcbbt = np.append(calibcbbt,xcalibcbbt)
            calibhbbt = np.append(calibhbbt,xcalibhbbt)
            ambPres = np.append(ambPres,xambPres)
            missingDataFlag = np.append(missingDataFlag,xmissingDataFlag)
    
    chsecs = np.copy(secs)
    mrad = mrad.T
    
    #I need to match the times of the AERI channel data with that from
    #the engineering file (which is the summary file).
    
    flag, prob = Other_functions.matchtimes(engsecs, chsecs, 0.5)
    if prob == 0:
        print('Error in read_aeri_ch: matchtimes() failed very badly')
        return err
    
    foo = np.where(flag == 1)[0]
    if len(foo) == 0:
        print('Error in read_aeri_ch: None of the ch data match the eng data times')
        return err
    
    chsecs = np.copy(chsecs[foo])
    mrad = np.copy(mrad[:,foo])
    hatchOpen = np.copy(hatchOpen[foo])
    missingDataFlag = np.copy(missingDataFlag[foo])
    bbsupport = np.copy(bbsupport[foo])
    calibambt = np.copy(calibambt[foo])
    calibcbbt = np.copy(calibcbbt[foo])
    calibhbbt = np.copy(calibhbbt[foo])
    ambPres = np.copy(ambPres[foo])
    
    #And now the reverse
    flag, prob = Other_functions.matchtimes(chsecs,engsecs,0.5)
    if prob == 0:
        print('Error in read_aeri_ch: matchtimes() failed very badly')
        return err

    foo = np.where(flag == 1)[0]
    if len(foo) == 0:
        print('Error in read_aeri_ch: None of the eng data match the ch data times')
        return err
    engsecs = np.copy(engsecs[foo])
    engtemp = np.copy(engtemp[foo])
    bbcavfactor = np.copy(bbcavfactor[foo])
    
    #Convert the time to something useful
    yy = np.array([datetime.utcfromtimestamp(x).year for x in chsecs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in chsecs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in chsecs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in chsecs])
    
    #I can only apply one of the Fv or Fa corrections, so abort if they are
    #both turned on
    
    if ((fv > 0) & (fa > 0)):
        print('Error: both the obscuration (Fv) and aft-optics (Fa) corrections are turned on')
        return err
    
    #Recalibrate the AERI data to ensure that the BB emissivity is correct.
    #Note that the typical (and now obsolete) value being used was 12.79
    #but we should be using something in excess of 35 (more like 39)
    
    if verbose == 3:
        print('Correcting for the blackbody emissivity cavity factor')
    nrad = mrad * 0
    emisn = Other_functions.get_aeri_bb_emis(wnum,39.0)
    for i in range(len(hour)):
        emiso = Other_functions.get_aeri_bb_emis(wnum, bbcavfactor[i])
        nrad[:,i] = Other_functions.aeri_recal(wnum, mrad[:,i], calibhbbt[i], calibhbbt[i], calibcbbt[i],
                    calibcbbt[i], calibambt[i], calibambt[i], emiso, emisn, emiso, emisn)
 
    mrad = np.copy(nrad)
    
    # If the Fa value is greater than zero, remove the contribution from the 
    # aft optics, but first see if the field "interferometerSecondPortTemp"
    # existed in the data, because if it did not then we can not apply this
    # correction and need to make some noise about it.
    
    foo = np.where(engtemp > 0)
    if ((len(foo) == 0) & (fa > 0)):
        print('Error: The algorithm wants to apply an aft-optics correction to the AERI')
        print('          data (i.e., fa > 0), but the field "interferometerSecondPortTemp"')
        print('          was not found in the input data. Thus we must abort this')
        print('          processing. Please either rerun with fa == 0 or modify the')
        print('          code so that a different field is used as the aft optic temperature')
        return err
    
    if fa > 0:
        if verbose >= 2:
            print('Correcting AERI data with Fa = ' + str(fa))
        nrad = np.copy(mrad)
        for i in range(len(hour)):
            Bref = Calcs_Conversions.planck(wnum,calibambt[i])
            Bcbb = Calcs_Conversions.planck(wnum,calibcbbt[i])
            Rc = emisn*Bcbb + (1.-emisn)*Bref
            aft_temp = np.interp(bt+to[i],engsecs,engtemp)
            
            #Apply a bit of QC here!
            if ((calibcbbt[i] - 20 > aft_temp) | (aft_temp > calibhbbt[i]+2)):
                print('Error: the temperature used for the aft optics (Fa correction) is out of bounds')
                return err
            Baft = Calcs_Conversions.planck(wnum,aft_temp)
            nrad[:,i] = mrad[:,i] - fa*Baft + fa*(mrad[:,i]-Rc)
        mrad = np.copy(nrad)
    
    if fv > 0:
        if verbose >= 2:
            print('Correcting AERI data with Fv = ' + str(fv))
        nrad = np.copy(mrad)
        for i in range(len(hour)):
            brad = Calcs_Conversions.planck(wnum, bbsupport[i])
            nrad[:,i] = (mrad[:,i] - fv*brad) / (1.-fv)
        mrad = np.copy(nrad)
    
    return ({'success':1, 'secs':chsecs, 'ymd':ymd, 'yy':yy, 'mm':mm, 'dd':dd,
            'hour':hour, 'wnum':wnum, 'rad':mrad, 'hatchopen':hatchOpen,
            'atmos_pres':ambPres,'missingDataFlag':missingDataFlag, 'fv':fv, 'fa':fa})


################################################################################
# This function controls the reading of the AERI, MWR, and ceilometer data. The function
# calls out many other functions to do this.
################################################################################
             
def read_all_data(date, retz, tres, dostop, verbose, avg_instant, ch1_path,
              pca_nf, fv, fa, sum_path, eng_path, aeri_type, cal_aeri_pres,
              aeri_smooth_noise, get_aeri_missingDataFlag, aeri_min_675_tb,
              aeri_max_675_tb, mwr_path, mwr_rootname, mwr_type, mwr_elev_field,
              mwr_n_tb_fields, mwr_tb_replicate, mwr_tb_field_names, mwr_tb_freqs,
              mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax, mwr_pwv_field,
              mwr_pwv_scalar, mwr_lwp_field, mwr_lwp_scalar, vceil_path, vceil_type,
              vceil_window_in, vceil_window_out, vceil_default_cbh, 
              hatchOpenSwitch, missingDataFlagSwitch, vip):

    fail = 0
    
    # Check the flag to make sure it has reasonable values
    if ((avg_instant != 0) & (avg_instant != 1)):
        print('Error: The "avg_instant" flag can only have a value of 0 (average) or 1 (instantaneous); do not average')
        fail = 1
        return fail, -999, -999, -999
        
    # Check the aeri_type flag. If it is -1, then we will not read any AERI data
    # in, and instead use MWR data as the master dataset. I will also simulate a
    # few channels on the AERI because it makes the rest of the code easier
    # (fewer changes) to make
    
    if ((aeri_type <= -1) & (mwr_type > 0)):
        
        #In this example, the "replicate" keyword should be unity. Since we aren't
        # using AERI data in this case, it should not have any other value
    
        if mwr_tb_replicate != 1:
            print('Error: The mwr_tb_replicate should be unity in this MWR-only retrieval (no AERI data)')
            fail = 1
            return fail, -999, -999, -999
    
        # I will read in the MWR data here and again later. I need to read it here
        # so that I can simulate AERI data (all as MISSING) so that the rest of the code
        # code can work.  Note that AERI_type (which is a negative number) also
        # defines the "step" used to read in the MWR data, which is pretty useful if
        # we are reading in HATPRO data which can have >3000 samples per day (i.e., the
        # aeri_type is used to subsample the MWR data)
        
        print(' Attempting to use MWR as the master instrument; no AERI data will be read in')
        
        mwr_data = read_mwr(mwr_path, mwr_rootname, date, mwr_type, abs(aeri_type), mwr_elev_field, mwr_n_tb_fields,
                           mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax,
                           mwr_pwv_field, mwr_pwv_scalar, mwr_lwp_field, mwr_lwp_scalar,
                           verbose, single_date=True)
        
        if mwr_data['success'] != 1:
            print('Problem reading MWR data -- unable to continue because the MWR is the master instrument')
            fail = 1
            return fail, -999, -999, -999
        else:
            #Quick check of the surface pressure field. If the values are negative,
            #then the code is not finding the pressure field and the retrieval won't
            #be able to run -- abort here.
            
            foo = np.where(mwr_data['psfc'] > 0)
            if len(foo) == 0:
                print('Error: Flags set to use MWR as master instrument, but Psfc not found -- must abort')
                fail = 1
                if dostop:
                    wait = input('Stopping inside routine for debugging. Press enter to continue')
                return fail, -999, -999, -999
            
            wnum = np.arange(int((950-900)/0.5)+1)*0.5+900            #Simulated wavenumber array
            mrad = np.ones((len(wnum),len(mwr_data['secs'])))*-999.0   #Radiance is all missing
            noise = np.ones((len(wnum),len(mwr_data['secs'])))         #Set all noise values to 1
            yy = np.array([datetime.utcfromtimestamp(x).year for x in mwr_data['secs']])
            mm = np.array([datetime.utcfromtimestamp(x).month for x in mwr_data['secs']])
            dd = np.array([datetime.utcfromtimestamp(x).day for x in mwr_data['secs']])
            hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in mwr_data['secs']])
            ymd = yy*10000 + mm*100 + dd

            aerieng = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'hour':hour,
                       'bbcavityfactor': np.zeros(len(mwr_data['secs'])),
                       'interferometerSecondPortTemp': np.ones(len(mwr_data['secs']))*300.0})
                       
            aerich1 = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'yy':yy, 'mm':mm, 'dd':dd, 'hour':hour,
                        'wnum':wnum, 'rad':mrad, 'hatchopen': np.ones(len(mwr_data['secs'])),
                        'atmos_pres':mwr_data['psfc'], 'missingDataFlag': np.zeros(len(mwr_data['secs'])),
                        'fv':0.0, 'fa': 0.0})
                        
            aerisum = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'hour':hour, 'wnum':wnum, 'noise':noise,
                        'lat':mwr_data['lat'], 'lon':mwr_data['lon'], 'alt':mwr_data['alt']})
        
    else:
        
        # Read in the AERI data
        
        aerieng = read_aeri_eng(eng_path,date,aeri_type,verbose)
        if aerieng['success'] == 0:
            print('Problem reading AERI eng data')
            if dostop:
                wait = input('Stopping inside routine for debugging. Press enter to continue')
            fail = 1
            return fail, -999, -999, -999
        
        aerich1 = read_aeri_ch(ch1_path,date,aeri_type,fv,fa,aerieng['secs'],
                              aerieng['interferometerSecondPortTemp'],
                              aerieng['bbcavityfactor'], get_aeri_missingDataFlag, verbose)
        
        aerisum = read_aeri_sum(sum_path,date,aeri_type,aeri_smooth_noise,verbose)
        
        if aerich1['success'] != 1:
            print('Problem reading AERI ch1 data')
            fail = 1
            return fail, -999, -999, -999
        if aerisum['success'] != 1:
            print('Problem reading AERI sum data')
            fail = 1
            return fail, -999, -999, -999
            
        #Apply the additional AERI QC tests.
        foo = np.where((aerich1['wnum'] >= 675) & (aerich1['wnum'] < 680))[0]
        if len(foo) == 0:
            for i in range(len(aerich1['secs'])):
                tmp = np.nanmean(aerich1['rad'][foo,i])
                tb = Calcs_Conversions.invplanck(677.5,tmp)
                if ((tb < aeri_min_675_tb) | (tb > aeri_max_675_tb)):
                    aerich1['missingDataFlag'][i] = 10
        
        # If the AERI data is noise filtered then I need to scale the AERI noise spectrum.
        # The scaling is wavenumber dependent, and I need to find a function that does this well...
        
        if pca_nf > 0:
            print('The AERI data have been PCA noise filtered. Someone needs to scale the AERI noise spectrum (not yet implimented)')
        
        if ((fail == 1) & (dostop != 0)):
            wait = input('Stopping inside routine for debugging. Press enter to continue')
        elif fail == 1:
            return fail, -999, -999, -999
        
        #Calibrate the AERI pressure sensor using a linear function
        if len(cal_aeri_pres) != 2:
            print('The calibration information for the AERI pressure sensor is ill-formed')
            fail = 1
        else:
            aerich1['atmos_pres'] = cal_aeri_pres[0] + aerich1['atmos_pres'] *cal_aeri_pres[1]
            
        if ((fail == 1) & (dostop != 0)):
            wait = input('Stopping inside routine for debugging. Press enter to continue')
        elif fail == 1:
            return fail, -999, -999, -999
    
    #Specify the times of the retrieved output
    if tres <= 0:
        ret_secs = aerich1['secs']-0.001     # Essentially the same as AERI sample time
        ret_tavg = 1./60                    # 1-s resolution
    else:
        yy = np.copy(aerich1['yy'])
        mm = np.copy(aerich1['mm'])
        dd = np.copy(aerich1['dd'])
        hour = np.copy(aerich1['hour'])
        if np.nanmax(hour) <= 24:
            nmins = 1440.0
        elif np.nanmax(hour) <= 2*24:
            nmins = 2*1440.0
        elif np.nanmax(hour) <= 3*24:
            nmins = 3*1440.0
        else:
            print('Error -- the AERI data files span more than 3 days -- code needs an update')
            fail = 1
            return fail, -999, -999, -999
        d = datetime(yy[0],mm[0],dd[0],0,0,0)
        bt = calendar.timegm(d.timetuple())
        ret_secs = np.arange(int(nmins/tres + 0.5)) * tres * 60+bt+tres/2.
        ret_tavg = tres
    
    #Read in the MWR zenith data
    mwr_data = read_mwr(mwr_path, mwr_rootname, date, mwr_type, 1, mwr_elev_field, mwr_n_tb_fields,
                        mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax,
                        mwr_pwv_field, mwr_pwv_scalar, mwr_lwp_field, mwr_lwp_scalar,
                        verbose)
                        
    if mwr_data['success'] != 1:
        print('Problem reading in MWR-zenith data')
    elif mwr_data['type'] > 0:
        print('Reading in MWR-zenith data')
        
    #Read in the MWR scan data
    mwrscan_data = read_mwrscan(vip['mwrscan_path'], vip['mwrscan_rootname'], date, vip['mwrscan_type'],
                   vip['mwrscan_elev_field'], vip['mwrscan_n_tb_fields'], vip['mwrscan_tb_field_names'],
                   vip['mwrscan_tb_freqs'], vip['mwrscan_tb_noise'], vip['mwrscan_tb_bias'],
                   vip['mwrscan_tb_field1_tbmax'], vip['mwrscan_n_elevations'], vip['mwrscan_elevations'], verbose)
                   
    if mwrscan_data['success'] == 0:
        print('Problem reading MWR-scan data')
        fail = 1
        return fail, -999, -999, -999
    elif mwrscan_data['type'] > 0:
        print('Reading in MWR-scan data')
    
    #Read in the ceilometer data
    vceil = read_vceil(vceil_path, date, vceil_type, ret_secs, verbose)
    
    if vceil['success'] < 0:
        fail = 1
    elif vceil['success'] == 0:
        if verbose >= 2:
            print('Problem reading vceil dat -- assuming ceilometer reported clear entire time')
        vceil = {'success':2, 'secs':aerich1['secs'], 'ymd':aerich1['ymd'], 'hour':aerich1['hour'], 'cbh':np.ones(len(aerich1['secs']))*-1}
    
    if ((fail == 1) & (dostop)):
        wait = input('Stopping for debugging. Press enter to continue')
    elif fail == 1:
        return fail, -999, -999, -999
        
    # Now apply a temporal screen to see if there are cloudy samples in the
    # AERI data by looking at the standard deviation of the 11 um data. If there
    # are cloudy samples, use the lidar to get an estimate of the cloud
    # base height for the subsequent retrieval
    
    caerich1 = Other_functions.find_cloud(aerich1, vceil, vceil_window_in, vceil_window_out, vceil_default_cbh)
    
    # Now put the AERI and MWR data on the same temporal grid.  Realize
    # that the RL sample time is for the start of the period, whereas the
    # AERI sample time is the middle of its period and the MWR's is the end.
    
    aeri = grid_aeri(caerich1, aerisum, avg_instant, hatchOpenSwitch, missingDataFlagSwitch,
                    ret_secs, ret_tavg, verbose)
    
    if aeri['success'] == 0:
        fail = 1
        return fail, -999, -999, -999
                    
    mwr = grid_mwr(mwr_data, avg_instant, ret_secs, ret_tavg, verbose)
    
    if mwr['success'] == 0:
        fail = 1
        return fail, -999, -999, -999
        
    mwrscan = grid_mwrscan(mwrscan_data, ret_secs, vip['mwrscan_n_elevations'],
                        vip['mwrscan_elevations'], vip['mwrscan_time_delta'], verbose)
                        
    if mwrscan['success'] == 0:
        fail = 1
        return fail, -999, -999, -999
        
    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')
    
    return fail, aeri, mwr, mwrscan

################################################################################
# This function is the one that actually reads in the MWR data.
################################################################################

def read_mwr(path, rootname, date, mwr_type, step, mwr_elev_field, mwr_n_tb_fields,
            mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias,
            mwr_tb_field1_tbmax, mwr_pwv_field, mwr_pwv_scalar, mwr_lwp_field,
            mwr_lwp_scalar, verbose, single_date=False):
            
    if verbose >= 2:
        print('Reading MWR data in ' + path)
    err = {'success':0, 'type':-1}
        
    # Check to make sure "step" is an integer greater than 0
    if(type(step) != int): 
        print("        Error: the step in read_mwr is not an integer -- aborting")
        return err
    elif(step < 0):
        print("        Error: the step in read_mwr is not a positive integer -- aborting")
        return err
        
        
    # If single date is True, only read in the single date. This is used for MWR only mode upon the first
    # read of the mwr data. This is mainly to get the times. If we read in data from before and after the
    # inputted date, it messes up the times.
    if single_date:
        udate = [str(date)]
    else:
        udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
                  str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]
              
    if mwr_type < 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "mwr_type" *******')
        print('   mwr_type=0 --->')
        print('             No MWR data specified')
        print('   mwr_type=1 --->')
        print('             MWR Tb data are in individual 1-d variables in the input file')
        print('   mwr_type=2 --->')
        print('             MWR tb data are in a single 2-d variable in the input file')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':-1}
        return err
        
    # Let's get the Tb frequencies, noise levels, and bias from the VIP file into arrays
    if mwr_n_tb_fields > 0:
        parts = mwr_tb_freqs.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR frequencies does not match desired MWR channels')
            return err
        else:
            freq = np.array(parts).astype(np.float)
        foo = np.where(((freq >= 15) & (freq < 240)))
        if len(parts) != mwr_n_tb_fields:
            print('Error: The entered frequencies are not between 15 and 240 GHz -- probably an error')
            return err
        parts = mwr_tb_bias.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR Tb bias values does not match desired MWR channels')
            return err
        else:
            bias = np.array(parts).astype(np.float)
        parts = mwr_tb_noise.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR Tb noise values does not match desired MWR channels')
            return err
        else:
            noise = np.array(parts).astype(np.float)
        
        # Only read MWR data if mwr_type > 0
        
    files = []
    if mwr_type > 0:
        for i in range(len(udate)):
            files = files + (glob.glob(path + '/' + '*' + rootname + '*' + udate[i] + '*cdf'))
        if len(files) == 0:
            print('Warning: No MWR files found for this date')
            mwr_type = 0
        
    # Only read MWR data if mwr_type > 0
    # Note that it is possible to read in MWR data but no Tbsky data (only PWV/LWP)
    nmwr_points = 0
    if mwr_type > 0:
        for i in range(len(files)):
            if verbose >= 3:
                print("Reading: " + files[i])
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            to = fid.variables['time_offset'][:]
            if len(to) <= 1:
                fid.close()
                continue
            lat = fid.variables['lat'][:]
            lon = fid.variables['lon'][:]
            alt = fid.variables['alt'][:]
                
            # Get the surface pressure field, if it exists
            # It might be "p_sfc" or "sfc_p" or "sfc_pres". Units are mb in both cases
            foo = np.where(np.array(list(fid.variables.keys())) == 'sfc_pres')[0]
            if len(foo) > 0:
                psfcx = fid.variables['sfc_pres'][:]
            else:
                foo = np.where(np.array(list(fid.variables.keys())) == 'sfc_p')[0]
                if len(foo) > 0:
                    psfcx = fid.variables['sfc_p'][:]
                else:
                   foo = np.where(np.array(list(fid.variables.keys())) == 'p_sfc')[0] 
                   if len(foo) > 0:
                       psfcx = fid.variables['p_sfc'][:]
                   else:
                       psfcx = np.ones(to.shape)*-999.0
            
            # See if the elevation variable exists. If so, read it in. If not then
            # assume all samples are zenith pointing and create the elev field as such
            foo = np.where(np.array(list(fid.variables.keys())) == mwr_elev_field)[0]
            if len(foo) > 0:
                elevx = fid.variables[mwr_elev_field][:]
            else:
                print('Warning: Unable to find the field ' + mwr_elev_field + ' in the MWR input file')
                elevx = np.ones(to.shape)*90.0
                
            # Read in the PWV and LWP fields in the input MWR file. If the field 
            # is not found, then assume the field is full of missing values.
            foo = np.where(np.array(list(fid.variables.keys())) == mwr_pwv_field)[0]
            if len(foo) == 0:
                print('Warning: Unable to find the PWV field ' + mwr_pwv_field + ' in the MWR input file')
                print(files[i])
                pwvx = np.ones(to.shape)*-999.0
            else:
                pwvx = fid.variables[mwr_pwv_field][:]
                pwvx = pwvx * mwr_pwv_scalar
            
            foo = np.where(np.array(list(fid.variables.keys())) == mwr_lwp_field)[0]
            if len(foo) == 0:
                print('Warning: Unable to find the LWP field ' + mwr_lwp_field + ' in the MWR input file')
                lwpx = np.ones(to.shape)*-999.0
            else:
                lwpx = fid.variables[mwr_lwp_field][:]
                lwpx = lwpx * mwr_pwv_scalar
            
            # Now read in the Tb data, if desired
            if mwr_n_tb_fields > 0:
                if mwr_type == 1:
                    
                    #If mwr_type is 1, then I expect to have mwr_n_tb_fields specified
                    fields = mwr_tb_field_names.split(',')
                    if len(fields) != mwr_n_tb_fields:
                        print('Error: The number of desired MWR Tb fields does not equal the number of input field names')
                        return err
                    
                    for j in range(mwr_n_tb_fields):
                        foo = np.where(np.array(list(fid.variables.keys())) == fields[j])[0]
                        if len(foo) == 0:
                            print('Error: Unable to find the Tb field ' + fields[j] + ' in the MWR input file')
                        tmp = fid.variables[fields[j]][:]
                        if j == 0:
                            tbskyx = np.copy(tmp)
                        else:
                            tbskyx = np.vstack((tbskyx,tmp))
                            
                elif mwr_type == 2:
                    
                    # if mwr_type is 2, then I expect there to only be a single mwr_tb_field_name
                    foo = np.where(np.array(list(fid.variables.keys())) == mwr_tb_field_names)[0]
                    if len(foo) == 0:
                        print('Error: Unable to find the Tb field ' + mwr_tb_field_names + ' in the MWR input file')
                        return err
                    tbskyx = fid.variables[mwr_tb_field_names][:].T
                    
                    # Now get the frequency field that goes with this 2-d array, as we need
                    # to select the subset of channels that are desired (i.e. probably not all
                    # of the channels in the dataset are desired). Select the channels that are
                    # closest in frequency to the entered (desired) frequencies.
                    
                    foo = np.where(np.array(list(fid.variables.keys())) == 'freq')[0]
                    if len(foo) == 0:
                        foo = np.where(np.array(list(fid.variables.keys())) == 'frequency')[0]
                        
                        if len(foo) == 0:
                            print('Error: Unable to find the field "freq" or "frequency" in the MWR input file')
                            return err
                        else:
                            freqx = fid.variables['frequency'][:]
                    else:
                        freqx = fid.variables['freq'][:]
                    
                    idx = np.ones(freq.shape)*-1
                    for j in range(len(freq)):
                        dell = abs(freq[j]-freqx)
                        foo = np.where(dell == np.nanmin(dell))[0]
                        idx[j] = foo
                    idx = idx.astype(int)
                    tbskyx = np.copy(tbskyx[idx,:])
                 
                else:
                    print('Error: Undefined mwr_type in VIP file -- aborting')
                    return err
            fid.close()
            
            # Now append the data (i.e., merge data from multiple files)
            
            if nmwr_points == 0:
                secs = bt+to
                elev = np.copy(elevx)
                if mwr_n_tb_fields > 0:
                    tbsky = np.copy(tbskyx)
                pwv = np.copy(pwvx)
                lwp = np.copy(lwpx)
                psfc = np.copy(psfcx)
            else:
                secs = np.append(secs, bt+to)
                elev = np.append(elev, elevx)
                if mwr_n_tb_fields > 0:
                    tbsky = np.append(tbsky,tbskyx,axis =1)
                pwv = np.append(pwv,pwvx)
                lwp = np.append(lwp,lwpx)
                psfc = np.append(psfc, psfcx)
            nmwr_points = len(secs)
        
        if len(secs) == 0:
            mwr_type = 0
        
        if mwr_type > 0:
            
            # Sanity check
            
            if mwr_n_tb_fields > 0:
                if len(tbsky[:,0]) != mwr_n_tb_fields:
                    print('Big problem in read_mwr')
                    return err
                tbsky0 = np.copy(tbsky)
                for i in range(mwr_n_tb_fields):
                    tbsky[i,:] = tbsky[i,:] + bias[i]
            
            # Select only the data that have an elevation of 90 degrees (zenith) and
            # where the bias-corrected brightness temperatures are above the cosmic background
            
            if mwr_n_tb_fields > 0:
                foo = np.where((((elev >= 89) & (elev < 91)) & ((tbsky[0,:] >= 2.7) & (tbsky[0,:] < mwr_tb_field1_tbmax))))[0]
            else:
                foo = np.where(((elev >= 89) & (elev < 91)))[0]
                
            if len(foo) == 0:
                print('Warning: All MWR data are at elevations other than 90 degrees (zenith)')
                mwr_type = 0
            else:
                secs = secs[foo]
                if mwr_n_tb_fields > 0:
                    tbsky = np.copy(tbsky[:,foo])
                    tbsky0 = np.copy(tbsky0[:,foo])
                pwv = pwv[foo]
                lwp = lwp[foo]
                psfc = psfc[foo]
    
    if mwr_type == 0:
        return {'success':1, 'type':mwr_type}
    else:
        yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
        mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
        dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
        hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
        ymd = yy*10000 + mm*100 + dd
        idx = np.arange(0,len(secs)/step)*step
        
        if mwr_n_tb_fields == 0:
           return ({'success':1, 'secs':secs[idx], 'ymd':ymd[idx], 'hour':hour[idx], 'pwv':pwv[idx], 'lwp':lwp[idx],
                 'lat':lat, 'lon':lon, 'alt':alt, 'psfc':psfc[idx], 'n_fields':mwr_n_tb_fields,
                 'type':mwr_type, 'rootname':rootname})
        else:
           return ({'success':1, 'secs':secs[idx], 'ymd':ymd[idx], 'hour':hour[idx], 'pwv':pwv[idx], 'lwp':lwp[idx],
                 'lat':lat, 'lon':lon, 'alt':alt, 'psfc':psfc[idx], 'n_fields':mwr_n_tb_fields,
                 'type':mwr_type, 'rootname':rootname, 'tbsky_orig':tbsky0[:,idx], 'tbsky_corr':tbsky[:,idx],
                  'freq':freq, 'noise':noise, 'bias':bias}) 
                  
################################################################################
# This function reads in the mwr scan data.
################################################################################

def read_mwrscan(path, rootname, date, mwrscan_type, mwrscan_elev_field, mwrscan_n_tb_fields,
            mwrscan_tb_field_names, mwrscan_tb_freqs, mwrscan_tb_noise, mwrscan_tb_bias,
            mwrscan_tb_field1_tbmax, mwrscan_n_elevations, mwrscan_elevations, verbose):
            
    if verbose >= 2:
        print('Reading MWR-scan data in ' + path)
    err = {'success':0, 'type':-1}
    
    # Read in the data from yesterday, today, and tomorrow
    udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
              str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]
              
    if mwrscan_type < 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "mwrscan_type" *******')
        print('   mwrscan_type=0 --->')
        print('             No MWR-scan data specified')
        print('   mwrscan_type=1 --->')
        print('             MWR-scan Tb data are in individual 1-d variables in the input file')
        print('   mwrscan_type=2 --->')
        print('             MWR-scan tb data are in a single 2-d variable in the input file')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':-1}
        return err
        
    # Let's get the Tb frequencies, noise levels, and bias from the VIP file into arrays
    if mwrscan_n_tb_fields > 0:
        parts = mwrscan_tb_freqs.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan frequencies does not match desired number of channels')
            return err
        else:
            freq = np.array(parts).astype(np.float)
        foo = np.where(((freq >= 15) & (freq < 240)))
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The entered MWR-scan frequencies are not between 15 and 240 GHz -- probably an error')
            return err
        parts = mwrscan_tb_bias.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan Tb bias values does not match desired number of channels')
            return err
        else:
            bias = np.array(parts).astype(np.float)
        parts = mwrscan_tb_noise.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan Tb noise values does not match desired number of channels')
            return err
        else:
            noise = np.array(parts).astype(np.float)
    if mwrscan_n_elevations > 0:
        parts = mwrscan_elevations.split(',')
        if len(parts) != mwrscan_n_elevations:
            print('Error: The number of entered MWR-scan elevations does not match desired number of elevations')
            return err
        else:
            delev = np.array(parts).astype(np.float)
    else:
        print('Error: The number of entered MWR-scan elevations must be larger than 0')
        return err
        
    
    
    # Only read MWR-scan data if mwrscan_type > 0
    files = []
    if mwrscan_type > 0:
        for i in range(len(udate)):
            files = files + (glob.glob(path + '/' + '*' + rootname + '*' + udate[i] + '*cdf'))
        if len(files) == 0:
            print('Warning: No MWR files found for this date')
            mwrscan_type = 0
     
    # Only read MWR-scan data if mwrscan_type > 0       
    nmwrscan_points = 0
    if mwrscan_type > 0:
        for i in range(len(files)):
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            to = fid.variables['time_offset'][:]
            if len(to) <= 1:
                fid.close()
                continue
            lat = fid.variables['lat'][:]
            lon = fid.variables['lon'][:]
            alt = fid.variables['alt'][:]
            
            # See if the elevation variable exists. If so, read it in. If not, then
            # assume all samples are zenith pointing and create the elev field as such
            vid = np.where(np.array(list(fid.variables.keys())) == mwrscan_elev_field)[0]
            if len(vid) > 0:
                elevx = fid.variables[mwrscan_elev_field][:]
            else:
                print('Warning: Unable to find the field ' + mwrscan_elev_field + ' in the MWR-scan input file')
            
            # Now read in the Tb data, if desired
            if mwrscan_n_tb_fields > 0:
                if mwrscan_type == 1:
                    
                    #If mwrscam_type is 1, then I expect to have mwrscan_n_tb_fields specified
                    fields = mwrscan_tb_field_names.split(',')
                    if len(fields) != mwrscan_n_tb_fields:
                        print('Error: The number of desired MWR-scan Tb fields does not equal the number of input field names')
                        return err
                    
                    for j in range(mwrscan_n_tb_fields):
                        vid = np.where(np.array(list(fid.variables.keys())) == fields[j])[0]
                        if len(vid) == 0:
                            print('Error: Unable to find the Tb field ' + fields[j] + ' in the MWR-scan input file')
                        tmp = fid.variables[fields[j]][:]
                        if j == 0:
                            tbskyx = np.copy(tmp)
                        else:
                            tbskyx = np.vstack((tbskyx,tmp))
                            
                elif mwrscan_type == 2:
                    
                    # if mwrscan_type is 2, then I expect there to only be a single mwrscan_tb_field_name
                    vid = np.where(np.array(list(fid.variables.keys())) == mwrscan_tb_field_names)[0]
                    if len(vid) == 0:
                        print('Error: Unable to find the Tb field ' + mwrscan_tb_field_names + ' in the MWR-scan input file')
                        return err
                    tbskyx = fid.variables[mwrscan_tb_field_names][:].T
                    
                    # Now get the frequency field that goes with this 2-d array, as we need
                    # to select the subset of channels that are desired (i.e. probably not all
                    # of the channels in the dataset are desired). Select the channels that are
                    # closest in frequency to the entered (desired) frequencies.
                    
                    vid = np.where(np.array(list(fid.variables.keys())) == 'freq')[0]
                    if len(vid) == 0:
                        vid = np.where(np.array(list(fid.variables.keys())) == 'frequency')[0]
                        
                        if len(vid) == 0:
                            print('Error: Unable to find the field "freq" or "frequency" in the MWR-scan input file')
                            return err
                        else:
                            freqx = fid.variables['frequency'][:]
                    else:
                        freqx = fid.variables['freq'][:]
                    
                    idx = np.ones(freq.shape, dtype=int)*-1
                    for j in range(len(freq)):
                        dell = abs(freq[j]-freqx)
                        foo = np.where(dell == np.nanmin(dell))[0]
                        idx[j] = foo
                    tbskyx = np.copy(tbskyx[idx,:])
                 
                else:
                    print('Error: Undefined mwrscan_type in VIP file -- aborting')
                    return err
            fid.close()
            
            # Now append the data (i.e., merge data from multiple files)
            
            if nmwrscan_points == 0:
                secs = bt+to
                elev = np.copy(elevx)
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.copy(tbskyx)
            else:
                secs = np.append(secs, bt+to)
                elev = np.append(elev, elevx)
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.append(tbsky,tbskyx,axis =1)
            nmwrscan_points = len(secs)
            
        if len(secs) == 0:
            mwrscan_type = 0
        
        if mwrscan_type > 0:
            
            # Sanity check
            
            if mwrscan_n_tb_fields > 0:
                if len(tbsky[:,0]) != mwrscan_n_tb_fields:
                    print('Big problem in read_mwrscan')
                    return err
                tbsky0 = np.copy(tbsky)
                for i in range(mwrscan_n_tb_fields):
                    tbsky[i,:] = tbsky[i,:] + bias[i]
            
            # This should never happen. If it does assume an error and abort
            if mwrscan_n_tb_fields <= 0:
                print('Error: The number of Tb fields in MWR-scan was less than 1 -- aborting')
                return err
                
            # Select only the data that have elevations that match the desired elevations (within 1 deg)
            # and where the bias-corrected brightness temperature are above the cosmic background (2.7 K)
            
            keep_samples = np.zeros(len(secs))    # all zeros right now
            for i in range(mwrscan_n_elevations):
                foo = np.where((((elev >= delev[i]-1) & (elev <= delev[i]+1)) & ((tbsky[0,:] >= 2.7) & (tbsky[0,:] < mwrscan_tb_field1_tbmax))))[0]
                if len(foo) > 0:
                    keep_samples[foo] = 1
            foo = np.where(keep_samples == 1)[0]
            if len(foo) == 0:
                print('Warning: Did not find any MWR-scan data at the desired elevations')
                mwrscan_type = 0
            else:
                secs = np.copy(secs[foo])
                elev = np.copy(elev[foo])
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.copy(tbsky[:,foo])
                    tbsky0 = np.copy(tbsky0[:,foo])
    
    if mwrscan_type == 0:
        return {'success':1, 'type':mwrscan_type}
    else:
        yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
        mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
        dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
        hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
        ymd = yy*10000 + mm*100 + dd
        
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'elev':elev,
                'lat':lat, 'lon':lon, 'alt':alt, 'n_fields':mwrscan_n_tb_fields,
                'type':mwrscan_type, 'rootname':rootname, 'tbsky_orig':tbsky0, 'tbsky_corr':tbsky,
                'freq':freq, 'noise':noise, 'bias':bias})

        
    
################################################################################
# This function reads in the aeri_eng file.
################################################################################

def read_aeri_eng(path, date, aeri_type, verbose):
    
    if verbose >= 2:
        print('Reading aeri_eng data in ' + path)
    err = {'success':0}
    if aeri_type <= 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "aeri_type" ******')
        print('   aeri_type=1 --->')
        print('             ARM-style ch1/ch2/sum/eng AERI input')
        print('   aeri_type=2 --->')
        print('             SSEC dmv-ncdf converted netcdf files, with')
        print('                     _sum.cdf and C1_rnc.cdf endings')
        print('   aeri_type=3 --->')
        print('             SSEC dmv-ncdf converted netcdf files, with ')
        print('                     .SUM.cdf and C1.RNC,cdf endings')
        print('   aeri_type=4 --->')
        print('             ARM-style ch1/ch2/sum AERI datastream names, but all of the')
        print('                     engineering data is found in the summary file (dmv2ncdf)')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':0}
        return err
    
    elif aeri_type == 1:
        filename = path + '/' + '*aeri*eng*' + str(date) + '*cdf'
    elif aeri_type == 2:
        filename = path + '/' + '*' + str(date % 2000000) + '_sum.cdf'
    elif aeri_type == 3:
        filename = path + '/' + '*' + str(date % 2000000) + '.SUM.cdf'
    elif aeri_type == 4:
        filename = path + '/' + '*aeri*sum*' + str(date) + '*cdf'
    else:
        print('Error in read_aeri_eng: unable to decode aeri_type')
        return err
    
    if verbose == 3:
        print('Looking for AERI engineering data as ' + filename)
        
    files = glob.glob(filename)
    if len(files) == 0:
        print('Error: Unable to find any AERI engineering data - aborting')
        print('Set aeri_type=0 and rerun to compare format of AERI data to aeri_type options')
        return err
        
    for jj in range(len(files)):
        fid = Dataset(files[jj],'r')
        bt = fid['base_time'][:]
        to = fid['time_offset'][:]
        
        #I want to get the temperature of the plug in the 2nd input port of the 
        #interfermoeter. The name of this field was different for the v2 and v4
        #AERI systems, so I have to look for either field
        
        if len(np.where(np.array(list(fid.variables.keys())) == 'interferometerSecondPortTemp')[0]) > 0:
            xinterferometerSecondPortTemp = fid.variables['interferometerSecondPortTemp'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'airNearInterferometerTemp')[0]) > 0:
            xinterferometerSecondPortTemp = fid.variables['airNearInterferometerTemp'][:]
        else:
            xinterferometerSecondPortTemp = np.ones(len(to))*-999.0
        
        #If the airNearInterferometerTemp exists, I will get this too. It is not used.
        
        if len(np.where(np.array(list(fid.variables.keys())) == 'airNearInterferometerTemp')[0]) > 0:
            xairNearInterferometerTemp = fid.variables['airNearInterferometerTemp'][:]
        else:
            xairNearInterferometerTemp = np.ones(len(to))*-999.0
            
        if len(np.where(np.array(list(fid.variables.keys())) == 'BBcavityFactor')[0]) > 0:
            xbbcavityfactor = fid.variables['BBcavityFactor'][:]
        else:
            xbbcavityfactor = np.ones(len(to))*12.79
            
        fid.close()
        
        #Append the data from the different files into single arrays
        if jj == 0:
            secs = bt+to
            interferometerSecondPortTemp = np.copy(xinterferometerSecondPortTemp)
            airNearInterferometerTemp = np.copy(xairNearInterferometerTemp)
            bbcavityfactor = np.copy(xbbcavityfactor)
        else:
            secs = np.append(secs,bt+to)
            interferometerSecondPortTemp = np.append(interferometerSecondPortTemp, xinterferometerSecondPortTemp)
            airNearInterferometerTemp = np.append(airNearInterferometerTemp, xairNearInterferometerTemp)
            bbcavityfactor = np.append(bbcavityfactor, xbbcavityfactor)
        
    #If the interferometerSecondPortTemp are all negative (i.e., missing)
    #then replace these values with the airNearInterferometerTemp
    
    foo = np.where(interferometerSecondPortTemp > 50)
    if len(foo) == 0:
        interferometerSecondPortTemp = np.copy(airNearInterferometerTemp)
    
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
    
    ymd = yy*10000 + mm*100 + dd
    
    return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour,
            'bbcavityfactor':bbcavityfactor,
            'interferometerSecondPortTemp':interferometerSecondPortTemp})


     
################################################################################
# This function reads the aeri_sum file
################################################################################

def read_aeri_sum(path,date,aeri_type,smooth_noise,verbose):
    if verbose >= 2:
        print('Reading aeri_sum data in ' + path)
    err = {'success':0}
    if aeri_type <= 0:
        print(('This piece of code should not be exercised, as this option should ' +
               'be screened in read_aeri_eng() earlier'))
        return err
    elif ((aeri_type == 1) | (aeri_type == 4)):
        filename = path + '/' + '*aeri*sum*' + str(date) + '*cdf'
    elif aeri_type == 2:
        filename = path + '/' + '*' + str(date % 2000000) + '_sum.cdf'
    elif aeri_type == 3:
        filename = path + '/' + '*' + str(date % 2000000) + '.SUM.cdf'
    else:
        print('Error in read_aeri_sum: unable to decode aeri_type')
        return err 
        
    if verbose >= 3:
        print('Looking for AERI summary data as ' + filename)
        
    files = glob.glob(filename)
    if len(files) == 0:
        print('Error: Unable to find any AERI summary data - aborting')
    
    for jj in range(len(files)):
        fid = Dataset(files[jj],'r')
        bt = fid.variables['base_time'][:]
        to = fid.variables['time_offset'][:]
        if aeri_type == 1:
            wnum1 = fid.variables['wnumsum5'][:]
            wnum2 = fid.variables['wnumsum6'][:]
        else:
            wnum1 = fid.variables['wnum1'][:]
            wnum2 = fid.variables['wnum2'][:]
        
        if len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENCh1')[0]) > 0:
            xnoise1 = fid.variables['SkyNENCh1'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENch1')[0]) > 0:
            xnoise1 = fid.variables['SkyNENch1'][:]
        else:
            print('Error in read_aeri_sum: unable to find the SkyNENCh1 field')
            return err
            
        if len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENCh2')[0]) > 0:
            xnoise2 = fid.variables['SkyNENCh2'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENch2')[0]) > 0:
            xnoise2 = fid.variables['SkyNENch2'][:]
        else:
            print('Error in read_aeri_sum: unable to find the SkyNENCh2 field')
            return err
        
        if len(np.where(np.array(list(fid.variables.keys())) == 'lat')[0]) > 0:
            lat = fid.variables['lat'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'Latitude')[0]) > 0:
            lat = np.ma.median(fid.variables['Latitude'][:])
        else:
            lat = -999.0
        if len(np.where(np.array(list(fid.variables.keys())) == 'lon')[0]) > 0:
            lon = fid.variables['lon'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'Longitude')[0]) > 0:
            lon =  np.ma.median(fid.variables['Longitude'][:])
        else:
            lon = -999.0
        if len(np.where(np.array(list(fid.variables.keys())) == 'alt')[0]) > 0:
            alt = fid.variables['alt'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'Altitude')[0]) > 0:
            alt =  np.ma.median(fid.variables['Altitude'][:])
        else:
            alt = 0.0
        fid.close()
        
        #Append the data together into single arrays
        if jj == 0:
            secs = bt + to
            noise1 = np.copy(xnoise1)
            noise2 = np.copy(xnoise2)
        else:
            secs = np.append(secs,bt+to)
            noise1 = np.append(noise1,xnoise1,axis = 0)
            noise2 = np.append(noise2,xnoise2,axis = 0)
    
    noise1 = noise1.T
    noise2 = noise2.T
    
    foo = np.where(wnum2 >= np.nanmax(wnum1)+0.1)[0]
    wnum = np.append(wnum1,wnum2[foo])
    noise = np.append(noise1,noise2[foo,:], axis = 0)
    
    # If desired, smooth the random error in the AERI observations
    # over at temporal window. This is needed because the AERI derives 
    # an independent noise estimate for each sample, but sometimes the 
    # temporal variation in the noise can impact the retrieval in
    # negative ways
    
    if smooth_noise > 0:
        if verbose >= 2:
            print('Smoothing the AERI noise temporally')
            
            #Derive the number of points to use in the smoother. The
            #variable "smooth_noise" is the temporal window in minutes,
            #so I will select the median time offset between samples
            #(which is in seconds) to determine this value
            
        delt = np.nanmedian(secs[1:len(secs)] - secs[0:len(secs)-1])
        npts = int(60*smooth_noise/delt)
        if npts % 2 == 0:
            npts = npts + 1
        snoise = np.copy(noise)
        
        #Apply the temporal noise smoother
        for j in range(len(secs)):
            vals = j + (np.arange(npts)-npts/2)
            foo = np.where((vals >= 0) & (vals <= len(secs)))[0]
            vals = vals[foo]
            for i in range(wnum):
                snoise[i,j] = np.nanmedian(noise[i,vals])
        npts = npts/4 + 1
        for i in range(wnum):
            snoise[i,:], flag = Other_functions.smooth(snoise[i,:],npts)
            if flag == 0:
                print('Error in smooth function. Aborting.')
                return err
        noise = np.copy(snoise)
    
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
                
    return ({'success':1, 'secs':secs,'ymd':ymd, 'hour':hour, 'wnum':wnum,
             'noise':noise, 'lat':lat, 'lon':lon, 'alt':alt})
             
################################################################################
# This function read in the ceilometer/vertically pointing lidar data.
################################################################################

def read_vceil(path, date, vceil_type, ret_secs, verbose):
    if verbose >= 2:
        print('Reading ceilometer data in ' + path)
        
    # Read in the cloud base height data from yesterday, today, and tomorrow.
    udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]
            
    vdate = [str(int((datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')) % 1000000) ,
            str(date % 1000000), str(int((datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')) % 1000000) ]
    
    err = {'success':0}
    
    files = []
    if vceil_type <= 0:
        print(' ')
        print('------------------------------------------------------')
        print('****** options for the VIP parameter "chb_type" ******')
        print('   cbh_type=1 --->')
        print('             ARM-style Vaisala ceilometer input')
        print('                     Files named "*ceil*cdf" ')
        print('                     Reads field "first_cbh", which has units m AGL')
        print('   cbh_type=2 --->')
        print('             Greg Blumbergs simple tranlation of ASOS CBH dataset')
        print('                     Files named "*ceil*cdf" or "cbh*cdf" ')
        print('                     Reads field "cloudHeight", which has units km AGL')
        print('   cbh_type=3 --->')
        print('             CLAMPS Doppler lidar fixed point (zenith) data')
        print('                     Files named "*dlfp*cdf"')
        print('                     Reads field "cbh", which has units km AGL')
        print('   cbh_type=4 --->')
        print('             ARM Doppler lidar "dlprofwstats" data')
        print('             Files named "*dlprofwstats*cdf" ')
        print('             Reads field "dl_cbh", which has units m AGL')
        print('   cbh_type=5 --->')
        print('             JOYCE-style Vaisala ceilometer input')
        print('                     Files named "*ct25k*nc" ')
        print('                     Reads field "first_cbh", which has units m AGL')
        print('-------------------------------------------------------')
        print(' ')
        err = {'success':-1}
        return err
        
    elif ((vceil_type == 1) | (vceil_type == 5)):
        if verbose == 3:
            print('Reading in VCEIL data')
        for i in range(len(udate)):
            files = files + (glob.glob(path + '/' + '*ceil*' + udate[i] + '*.cdf'))
        if len(files) == 0:
            for i in range(len(udate)):
                files = files + (glob.glob(path + '/' + '*ceil*' + udate[i] + '*.nc'))
            if len(files) == 0:
                for i in range(len(vdate)):
                    files = files + (glob.glob(path + '/' + vdate[i] + '*ct25k*' + '*.cdf'))
                if len(files) == 0:
                    for i in range(len(vdate)):
                        files = files + (glob.glob(path + '/' + vdate[i] + '*ct25k*' + '*.nc'))
        if len(files) == 0:
            print('No CBH files found for this date')
            return err
        
        for i in range(len(files)):
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
                to = fid.variables['time_offset'][:]
            elif len(np.where(np.array(list(fid.variables.keys())) == 'time')[0]) > 0:
                to = fid.variables['time'][:]
            else:
                fid.close()
                print(' Error reading the time fields -- aborting read_vceil')
                return err
            cbhx = fid.variables['first_cbh'][:]
            fid.close()
            cbhx = cbhx / 1000.
        
            if i == 0:
                secs = bt+to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)
    
    elif vceil_type == 2:
        if verbose == 3:
            print('Reading in ASOS/AWOS')
            
        for i in range(len(udate)):
            files = files + (glob.glob(path + '/' + '*ceil*' + udate[i] + '*.cdf'))
        if len(files) == 0:
            for i in range(len(udate)):
                files = files + (glob.glob(path + '/' + '*cbh*' + udate[i] + '*.cdf'))
            if len(files) == 0:
                print('No CBH files found for this data')
                return err
        
        for i in range(len(files)):
            if verbose == 3:
                print(' Reading in file ' + files(i))
            
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            to = fid.variables['time_offset'][:]
            cbhx = fid.variables['cloudHeight'][:]
            fid.close()
            
            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)
    
    elif vceil_type == 3:
        if verbose == 3:
            print('Reading in CLAMPS dlfp data')
        for i in range(len(udate)):
            files = files + (glob.glob(path + '/' + '*dlfp*' + udate[i] + '*.cdf'))
        if len(files) == 0:
            print('No CBH files found for this date')
            return err
        for i in range(len(files)):
            if verbose == 3:
                print('Reading the file ' + files[i])
        
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            to = fid.variables['time_offset'][:]
            cbhx = fid.variables['cbh'][:]
            fid.close()
        
            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)
            
    elif vceil_type == 4:
        ftype = ['nc','cdf']
        if verbose == 3:
            print(' Reading in ARM dlprofwstats data')
        for i in range(len(udate)):
            for j in range(len(ftype)):
                files = files + (glob.glob(path +'/' + '*dlprofwstats*' + udate[i] + '*.' + ftype[j]))
        if len(files) == 0:
            print('No CBH files found for this date')
            return err
        
        for i in range(len(files)):
            if verbose == 3:
                print(' Reading the file ' + files[i])
            fid = Dataset(files[i],'r')
            bt = fid.variables['base_time'][:]
            to = fid.variables['time_offset'][:]
            cbhx = fid.variables['dl_cbh'][:]
            fid.close()
            
            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)
        cbh = cbh/1000.              # Convert m AGL to km AGL
    
    else:
        print('Error in read_vceil: Undefined ceilometer type')
        return err
    
    # Put in a trap for the GDL version of the code, as occassionally some
    # of Greg's mimiced ceilometer data files (generated from ASOS/AWOS data
    # have the base_time as type int64, which GDL did not handle well...
    
    if (bt == 0):
        print('Error in read_vceil: Unable to read in base_time -- likely the GDL/int64 issue')
        return err
    
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
    
    return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'cbh':cbh})
    
################################################################################
# This function puts the AERI data on the same temporal grid as the MWR data.
################################################################################

def grid_aeri(ch1, aerisum, avg_instant, hatchOpenSwitch, missingDataFlagSwitch,
              secs, tavg, verbose):
            
    if verbose == 3:
        print('Temporally gridding the AERI data')
    
    err = {'success':0}
    
    rrad = np.zeros((len(ch1['wnum']),len(secs)))
    nrad = np.zeros((len(aerisum['wnum']),len(secs)))
    cbh = np.zeros(len(secs))
    cbhflag = np.zeros(len(secs))
    hatflag = np.zeros(len(secs))
    mssflag = np.zeros(len(secs))
    atmos_pres = np.zeros(len(secs))
    
    for i in range(len(secs)):
        # Get the channel 1 data on this grid
        
        if ((hatchOpenSwitch == 1) & (missingDataFlagSwitch == 0)):
            if ((i == 0) & (verbose >= 2)):
                print('Only averaging AERI data where hatchOpen is 1 (missingDataFlag is anything)')
            if avg_instant == 0:
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
            else:
                dell = np.abs(secs[i] - ch1['secs'])
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
                
                if len(foo) > 1:
                    foo = np.array([foo[0]])
                    nfoo = 1
        elif ((hatchOpenSwitch == 0) & (missingDataFlagSwitch == 0)):
            if ((i == 0) & (verbose >= 2)):
                print('Averaging all AERI data regardless of hatchOpen or missingDataFlag')
            if avg_instant == 0:
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.))[0]
            else:
                dell = np.abs(secs[i] - ch1['secs'])
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)))[0]
                
                if len(foo) > 1:
                    foo = np.array([foo[0]])
                    nfoo = 1
        elif ((hatchOpenSwitch == 1) & (missingDataFlagSwitch == 1)):
            if ((i == 0) & (verbose >= 2)):
                print('Only averaging AERI data where hatchOpen is 1 and missingDataFlag is 0')
            if avg_instant == 0:
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
            else:
               dell = np.abs(secs[i] - ch1['secs'])
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0) & (dell == np.nanmin(dell)) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
                               
               if len(foo) > 1:
                    foo = np.array([foo[0]])
                    nfoo = 1
                    
        elif ((hatchOpenSwitch == 0) & (missingDataFlagSwitch == 1)):
            if ((i == 0) & (verbose >= 2)):
                print('Averaging all AERI data where missingDataFlag is 0 (hatchOpen can be anything)')
                
            if avg_instant == 0:
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0))[0]
            else:
               dell = np.abs(secs[i] - ch1['secs'])
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0) & (dell == np.nanmin(dell)))[0]
                               
               if len(foo) > 1:
                    foo = np.array([foo[0]])
                    nfoo = 1
                    
        else:
            print('This piece of code should never be executed -- logic trap in grid_aeri()')
            return err
        
        if len(foo) == 0:
            rrad[:,i] = -9999.
            cbhflag[i] = -9999
            hatflag[i] = -9999
            mssflag[i] = 1    #The sample is missing
            atmos_pres[i] = -9999.
        else:
            if len(foo) == 1:
                rrad[:,i] = np.squeeze(ch1['rad'][:,foo])
                cbh[i] = ch1['cbh'][foo]
                cbhflag[i] = ch1['cbhflag'][foo]
                hatflag[i] = ch1['hatchopen'][foo]
                mssflag[i] = ch1['missingDataFlag'][foo]
                atmos_pres[i] = ch1['atmos_pres'][foo]
            else:
                rrad[:,i] = np.nansum(ch1['rad'][:,foo], axis = 1)/np.float(len(foo))
                mssflag[i] = np.nanmax(ch1['missingDataFlag'][foo])
                
                # Determine the appropriate value for the cbh.  If one of the
                # "inners" are set, then use the median of these values.  Otherwise,
                # use the median of the "outers".  Last resort is to use the "default"

                bar = np.where(ch1['cbhflag'][foo] == 1)[0]    # Look for "inners"
                if len(bar) > 0:
                    if len(bar) == 1:
                        cbh[i] = ch1['cbh'][foo[bar]]
                    else:
                        cbh[i] = np.nanmedian(ch1['cbh'][foo[bar]])
                    cbhflag[i] = 1

                else:
                    bar = np.where(ch1['cbhflag'][foo] == 2)[0] # Look for "outers"
                    if len(bar) > 0:
                        if len(bar) == 1:
                            cbh[i] = ch1['cbh'][foo[bar]]
                        else:
                            cbh[i] = np.nanmedian(ch1['cbh'][foo[bar]])
                        cbhflag[i] = 2
                    else:
                        cbh[i] = np.nanmedian(ch1['cbh'][foo])
                        cbhflag[i] = 3

                # Determine the appropriate value for the hatch, given there
                # are several AERI samples in this window.  If the hatch is open
                # for all of these samples, then call it open.  If the hatch is
                # closed for all of the samples, then call it closed.  Otherwise
                # call it "indeterminant"

                bar = np.where(ch1['hatchopen'][foo] == 1)[0]
                if len(bar) == len(foo):
                    hatflag[i] = 1                             # Hatch was always open
                else:
                    np.where(ch1['hatchopen'][foo] == 0)[0]
                    if len(bar) == len(foo):
                      hatflag[i] = 0                           # Hatch was always closed
                    else:
                      hatflag[i] = 3                           # If we are here it is neither
            #print len(ch1['secs'])
            #print len(ch1['atmos_pres'])
            #print foo
            atmos_pres[i] = np.nanmean(ch1['atmos_pres'][foo])

        # Get the summary data on this grid
        if avg_instant == 0:
            foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.))[0]
        else:
            dell = np.abs(secs[i]-aerisum['secs'])
            foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)))[0]

        if len(foo) > 1:
            foo = np.array([foo[0]])
            nfoo = 1
      
        if len(foo) == 0:
            nrad[:,i] = -9999.
        elif len(foo) == 1:
            nrad[:,i] = np.squeeze(aerisum['noise'][:,foo])
        else:
            
            # Divide by sqrt N when averaging many samples together
            # In the end, I think that the noise spectrum is getting
            # too compressed using the correct way of dividing by
            # the sqrt(N).  However, I would like there to be some
            # noise compression when averaging data, so I am going to
            # use this 'hack'.

            #nrad[:,i] = (np.nansum(aerisum['noise'][:,foo],axis = 1)/np.float(len(foo))) / np.sqrt(len(foo))
            nrad[:,i] = (np.nansum(aerisum['noise'][:,foo],axis = 1)/np.float(len(foo))) / np.sqrt(len(foo))

    # Put all AERI data on same spectral grid
    wnum = np.copy(ch1['wnum'])
    noise = rrad*0.
    for i in range(len(secs)):
        noise[:,i] = np.interp(wnum,aerisum['wnum'],nrad[:,i])

        # Get the surface temperature from the AERI radiance observations
        # Use the actual AERI radiances, not the subset that was extracted

    bar = np.where((wnum >= 670) & (wnum <= 675))[0]

    if len(bar) >= 8:
        Tsfc = Calcs_Conversions.invplanck(np.nansum(wnum[bar])/np.float(len(bar)),np.nansum(rrad[bar,:],axis = 0)/np.float(len(bar)))
        Tsfc = Tsfc - 273.16

        bar = np.where(np.isnan(Tsfc))

        if len(bar) > 0:
            Tsfc[bar] = -999.
    else:
        Tsfc = np.ones(len(secs))*-999.

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'yy':yy, 'mm':mm, 'dd':dd,
            'cbh':cbh, 'cbhflag':cbhflag, 'hatchopen':hatflag, 'avg_instant':avg_instant,
            'wnum':wnum, 'radmn':rrad, 'noise':noise, 'atmos_pres':atmos_pres,
            'tsfc':Tsfc, 'fv':ch1['fv'], 'fa':ch1['fa'], 'missingDataFlag':mssflag,
            'lat':aerisum['lat'],'lon':aerisum['lon'],'alt':aerisum['alt']})
            
################################################################################
# This function puts the MWR data onto a common temporal grid.
################################################################################

def grid_mwr(mwr, avg_instant, secs, tavg, verbose):
    
    if verbose == 3:
        print('Temporally gridding the MWR data')
    err = {'success':0}
    
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
    
    if mwr['type'] <= 0:
        missing = np.ones(len(secs))*-999.
        return {'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'n_fields':0,
                'pwv':missing, 'lwp':missing, 'type':mwr['type'], 'rootname':'none found'}
    
    # If the Tavg is too low (or zero), then inflate it somewhat. Units are minutes
    
    if tavg < 2:
        print('Tavg for MWR data is too small. Setting it equal to 2 seconds.')
        twin = 2
    else:
        twin = tavg
        
    # Allocate the needed space
    pwv = np.ones(len(secs))*-999.
    lwp = np.ones(len(secs))*-999.
    
    if mwr['n_fields'] > 0:
        tbsky = np.ones((mwr['n_fields'],len(secs)))*-999.0
        
        # Grid the data
    if avg_instant == 0:
        # We are averaging the MWR data over the averaging interval
        
        for i in range(len(secs)):
            foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.) & (mwr['pwv'] > 0))[0]
            if len(foo) > 0:
                pwv[i] = np.nanmean(mwr['pwv'][foo])
                lwp[i] = np.nanmean(mwr['lwp'][foo])
            
            if mwr['n_fields'] > 0:
                foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0]
                
                if len(foo) > 0:
                    for j in range(mwr['n_fields']):
                        tbsky[j,i] = np.nanmean(mwr['tbsky_corr'][j,foo])
    
    elif avg_instant == 1:
        # We are taking the closest point to the center of the averaging interval,
        # but the point taken must be within the averaging interval
        
        for i in range(len(secs)):
            foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.) & (mwr['pwv'] > 0))[0]
            if len(foo) > 0:
                dell = np.abs(secs[i] - mwr['secs'][foo])
                bar = np.where(dell == np.nanmin(dell))[0][0]
                pwv[i] = mwr['pwv'][foo[bar]]
                lwp[i] = mwr['lwp'][foo[bar]]
            
            if mwr['n_fields'] > 0:
                foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0] 
                
                if len(foo) > 0:
                    dell = np.abs(secs[i] - mwr['secs'][foo])
                    bar = np.where(dell == np.nanmin(dell))[0][0]
                    for j in range(mwr['n_fields']):
                        tbsky[j,i] = mwr['tbsky_corr'][j,foo[bar]]
    else:
        print(' Error: the avg_instant flag has an unknown value in grid mwr()')
        return err
    
    # The structure being returned depends on the number of Tb fields desired
    
    if mwr['n_fields'] == 0:
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'pwv':pwv,
                'lwp':lwp, 'n_fields':0, 'type': mwr['type'], 'rootname':mwr['rootname']})
    else:
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'pwv':pwv,
                'lwp':lwp, 'n_fields':mwr['n_fields'], 'tbsky':tbsky, 'freq':mwr['freq'],
                'noise':mwr['noise'], 'bias':mwr['bias'], 'type': mwr['type'], 'rootname':mwr['rootname']})
                
################################################################################
# This function puts the mwrscan data onto a common temporal grid.
################################################################################

def grid_mwrscan(mwrscan, secs, n_elevations, elevations, timewindow, verbose):
    
    if verbose == 3:
        print(' Temporally gridding the MWR-scan data')
    err = {'success':0}
    
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
    
    if mwrscan['type'] <= 0:
        missing = np.ones(len(secs))*-999.
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'n_fields':0,
                'type':mwrscan['type'], 'rootname':'none found'})
    
    parts = elevations.split(',')
    
    if len(parts) != n_elevations:
        print('Error: Problem with elevations in grid_mwrscan (this should not happen)')
        return err
    else:
        delev = np.array(parts).astype(np.float)
        
    # Compute the time window to use in seconds, and it should be at least 2 minutes
    
    if ((timewindow*3600.) < (2.*60)):
        print('Original time window for averaging mwr scan data was too small. Changing to 2 minutes.')
        twin = 2*60.
    else:
        twin = timewindow*3600.
    
    # Allocate the needed space
    
    tbsky = np.ones((n_elevations,mwrscan['n_fields'],len(secs)))*-999.
    
    # Regardless of the averaging type (averaging or taking the closest one)
    # I am going to take the closest one at each elevation scan (if it is within 
    # the averaging window) and assign it to the sample time.
    
    for i in range(len(secs)):
        for j in range(n_elevations):
            foo = np.where((mwrscan['secs'] >= secs[i]-twin) & (mwrscan['secs'] < secs[i]+twin) &
                            (np.abs(mwrscan['elev'] - delev[j]) < 1))[0]
            
            if len(foo) > 0:
                dell = np.abs(secs[i] - mwrscan['secs'][foo])
                bar = np.where(dell == np.nanmin(dell))[0][0]
                tbsky[j,:,i] = mwrscan['tbsky_corr'][:,foo[bar]]
            else:
                tbsky[j,:,i] = -999.
                
    # Now I have the tbksy data in a 3-d structure, and I need to reshape it into
    # a 2-d structure (mixing frequency and elevation angle).  
    
    ntbsky = np.ones((n_elevations*mwrscan['n_fields'],len(secs)))*-999.
    dim = np.ones(n_elevations*mwrscan['n_fields'])*-999.
    noise = np.ones(n_elevations*mwrscan['n_fields'])*-999.
    idx = np.arange(mwrscan['n_fields'])
    for i in range(n_elevations):
        ntbsky[i*mwrscan['n_fields']+idx,:] = tbsky[i,:,:]
        dim[i*mwrscan['n_fields']+idx] = mwrscan['freq']*1000. + delev[i]/1000.
        noise[i*mwrscan['n_fields']+idx] = mwrscan['noise']

    return ({'success': 1, 'secs': secs, 'ymd': ymd, 'hour': hour, 'n_elevations': n_elevations,
            'elevations': delev, 'n_fields': mwrscan['n_fields'], 'tbsky': ntbsky, 'freq': mwrscan['freq'],
            'noise': noise, 'bias': mwrscan['bias'], 'dim': dim, 'type': mwrscan['type'],
            'rootname': mwrscan['rootname'], 'timewindow': timewindow})

################################################################################
# This function reads in the thermodynamic profiles from the external source
# (sonde, lidar, etc)
################################################################################

def read_external_profile_data(date, ht, secs, tres, avg_instant,
            wv_prof_type, wv_prof_path, wv_noise_multiplier_hts, wv_noise_multiplier_val,
            wv_prof_minht, wv_prof_maxht, wv_time_delta, temp_prof_type, temp_prof_path,
            temp_noise_adder_hts, temp_noise_adder_val, temp_prof_minht, temp_prof_maxht,
            temp_time_delta, dostop, verbose):
            
    external = {'success':0, 'nTprof':-1, 'nQprof':-1}
    
    model_type = 'None'
    model_lat = -999.
    model_lon = -999.
    
    #### Read the external water vapor data first
    
    # No external WV source specified....
    
    qunit = ' '
    wvmultiplier = 1.0              # I will need this below for the DIAL data
    
    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]
    
    cdf = ['nc','cdf']       
    if wv_prof_type == 0:
        a = 0           # Do nothing -- read nothing -- make no noise at all
        qtype = 'None'
        external['nQprof'] = 0.0
        
    # Read in  the ARM radiosondes as a constraint on the water vapor profile 
    # over some range
    
    elif wv_prof_type == 1:
        if verbose >= 1:
            print('Reading in ARM radiosonde data to contrain the WV profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + sorted(glob.glob(wv_prof_path + '/' + '*sonde*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM radiosondes found in this directory for this date')
            external['nQprof'] = 0.0
        else:
            maxht = int(wv_prof_maxht+0.1)
            if maxht < wv_prof_maxht:
                maxht += 1
            zzq = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]
            
            external['nQprof'] = 0
            for i in range(len(files)):
                 fid = Dataset(files[i],'r')
                 bt = fid.variables['base_time'][0]
                 to = fid.variables['time_offset'][:]
                 p = fid.variables['pres'][:]
                 t = fid.variables['tdry'][:]
                 u = fid.variables['rh'][:]
                 z = fid.variables['alt'][:]
                 
                 fid.close()
                 z = (z-z[0])/1000.
                 foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103))[0]
                 
                 if len(foo) < 2:
                     continue
                 z = z[foo]
                 p = p[foo]
                 t = t[foo]
                 u = u[foo]
                 if np.nanmax(z) < maxht:
                     continue            # The sonde must be above this altitude to be used here
                     
                 # Compute WVMR and its uncertainty (using RMS and typical sonde uncertainties
                 # of 3% in Rh and 0.5 degC in temperature)
                 
                 w = Calcs_Conversions.rh2w(t, u/100., p)
                 w1 = Calcs_Conversions.rh2w(t, u/100.-0.03, p)
                 w2 = Calcs_Conversions.rh2w(t+0.5, u/100., p)
                 we = np.sqrt((w - w1)**2. + (w-w2)**2)            # Sum of squared errors
                 qunit = 'g/kg'
                 qtype = 'ARM radiosonde'
                 
                 # Append the data to the growing structure
                 
                 if external['nQprof'] <= 0:
                     qsecs = bt + to[0]
                     wv = np.interp(zzq,z,w)
                     swv = np.interp(zzq,z,we)
                 else:
                     qsecs = np.append(qsecs,bt+to[0])
                     wv = np.vstack((wv, np.interp(zzq,z,w)))
                     swv = np.vstack((swv, np.interp(zzq,z,we)))
                 external['nQprof'] += 1
                 
            if external['nQprof'] > 0:
                wv = wv.T
                swv = swv.T
                
    # Read in the ARM Raman lidar data (rlprofmr)
    
    elif wv_prof_type == 2:
        if verbose >= 1:
            print('Reading in ARM Raman lidar (rlprofmr) data to constrain the WV profile')
        
        qunit = 'g/kg'
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(wv_prof_path + '/' + '*rlprofmr*' + dates[i] + '*.' + cdf[j]))
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM RLID WV found in this directory for this date')
            external['nQprof'] = 0.
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' ARM RLID WV data files')
            nprof = 0.
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                
                # There are two types of RLPROF_MR data.
                # This handles the rlprofmr1turn dataset.
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'mixing_ratio_3')[0]) > 0:
                    qtype = 'ARM Raman lidar (rlprofmr1turn)'
                    htx = fid.variables['height'][:]
                    wvx = fid.variables['mixing_ratio_3'][:]
                    swvx = fid.variables['mixing_ratio_3_error'][:]
                    qcflag = fid.variables['qc_flag'][:]
                    fid.close()
                    
                    # Only keep data above 70 m to ensure the 60-m tower isn't included
                    
                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('This should not happen when processing the RLID WV data')
                    htx = htx[foo]
                    wvx = wvx[foo,:]
                    swvx = wvx[foo,:]
                    
                    # And only keep samples where the qc_flag is 0 (i.e. good quality)
                    foo = np.where(qcflag == 0)[0]
                    if len(foo) == 0:
                        print('Warning: no good samples found for the RLID WV on this day')
                        continue
                    else:
                        to = to[foo]
                        wvx = wvx[:,foo]
                        swvx = swvx[:,foo]
                
                else:
                    # This handles the rlprofmr1news dataset
                    qtype = 'ARM Raman lidar (rlprofmr1news)'
                    htx = fid.variables['height'][:]
                    wvx = fid.variables['mr_merged'][:]
                    swvx = fid.variables['mr_merged_err'][:]
                    fid.close()
                    
                    #Only keep data above 70 m to ensure the 60-m tower isn't included
                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('This should not happen when processing the RLID WV data')
                    htx = htx[foo]
                    wvx = wvx[foo,:]
                    swvx = swvx[foo,:]
                    
                # Now append the data to the growing structure
                if nprof == 0:
                    qsecs = bt+to
                    wv = np.copy(wvx)
                    swv = np.copy(swvx)
                else:
                    qsecs = np.append(qsecs, bt+to)
                    wv = np.append(wv,wvx, axis = 1)
                    swv = np.append(swv,swvx, axis = 1)
                nprof = len(qsecs)
            
            external['nQprof'] = len(qsecs)
        
        if external['nQprof'] > 0:
            zzq = np.copy(htx)
            wvmultiplier = 1.      # To scale the WV profiles to be reasonable order of magnitude
            wv = wv/wvmultiplier
            swv = wv/wvmultiplier
        
        
    # Read in the NCAR WV DIAL data
    
    elif wv_prof_type == 3:
        if verbose >= 1:
            print('Reading in NCAR WV DIAL data to constrain the WV profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(wv_prof_path + '/' + '*' + str(int(dates[i])%1000000) + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No NCAR WV DIAL found in this directory for this date')
            external['nQprof'] = 0.0
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' NCAR WV DIAL data files')
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                secsx = fid.variables['time_unix'][:]
                zzq = fid.variables['range'][:]
                wvx = fid.variables['N_avg'][:]
                swvx = fid.variables['N_error'][:]
                fid.close()
                
                if i == 0:
                    qsecs = np.copy(secsx)
                    wv = np.copy(wvx)
                    swv = np.copy(swvx)
                else:
                    qsecs = np.append(qsecs,secsx)
                    wv = np.append(wv, wvx, axis = 1)
                    swv = np.append(swv, swvx, axis = 1)
                external['nQprof'] = len(qsecs)  
            zzq = zzq / 1000.           # Convert m AGL to km AGL
        
        if external['nQprof'] > 0:
            wvmultiplier = 1e16           # To scale the WV profiles to be reasonable order of magnitude
            wv = wv/wvmultiplier
            swv = swv/wvmultiplier
            qunit = 'molecules/cm3 (sccaled by 1e16)'
            qtype = 'NCAR water vapor DIAL'

        wv = wv.T
        swv = swv.T
    
    # Read in the numerical weather model soundings (Greg Blumberg's format)
    
    elif wv_prof_type == 4:
        if verbose >= 1:
            print('Reading in NWP model output to constrain the WV profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(wv_prof_path + '/' + '*model*' + dates[i] + '*.' + cdf[j]))
        if len(files) == 0:
            if verbose >= 1:
                print('No NWP model output data ound in this directory for this date')
            external['nQprof'] = 0
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files))  + ' NWP output WV files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                zzq = fid.variables['height'][:]
                wwx = fid.variables['waterVapor'][:]
                ssx = fid.variables['sigma_waterVapor'][:]
                model_type = fid.model
                model_lat = np.float(fid.gridpoint_lat)
                model_lon = np.float(fid.gridpoint_lon)
                fid.close()
                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.vstack((wv, wwx))
                    swv = np.vstack((swv,ssx))
                    
            wv = wv.T
            swv = swv.T
            external['nQprof'] = len(qsecs)
            
            # Perform a quick check to make sure all profiles exist. I am
            # assuming that (since this is a model) I only have to look at the 
            # first level. If there are no valid profiles (ie, all of them are
            # NaN, then reset the counter to zero
            
            foo = np.where(((~np.isnan(wv[0,:])) & (np.abs(wv[0,:] < 500))))[0]
            if len(foo) > 0:
                wv = wv[:,foo]
                swv = swv[:,foo]
                qsecs = qsecs[foo]
                external['nQprof'] = len(qsecs)
            else:
                external['nQprof'] = 0
            
        qunit = 'g/kg'
        qtype = 'NWP model output from ' + model_type + ' at ' + str(model_lat) + ' degN, ' + str(model_lon) + ' degE'
        
    # Read in the Vaisala water vapor DIAL profiles
    elif wv_prof_type == 5:
        if verbose >= 1:
            print('Reading in Vaisala WV DIAL to constrain the WV profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(wv_prof_path + '/' + '*dial*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No Vaisala WV DIAL data found in this directory for this date')
            external['nQprof'] = 0
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' Vaisala WV DIAL files')
            
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                zzq = fid.variables['height'][:]
                wwx = fid.variables['waterVapor'][:]
                ssx = fid.variables['sigma_waterVapor'][:]
                mhtx = fid.variables['maxht'][:]
                fid.close()
                
                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                    maxht = np.copy(mhtx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,wwx, axis = 1)
                    swv = np.append(swv,ssx, axis = 1)
            
            external['nQprof'] = len(qsecs)
            
            # Set all of the points above the maximum height to -999.
            
            for i in range(len(qsecs)):
                foo = np.where(zzq >= maxht[i])[0]
                if len(foo) > 0:
                    wv[foo,i] = -999.
                    swv[foo,i] = -999.
            
        qunit = 'g/kg'
        qtype = 'Vaisala WV DIAL data'
        
    elif wv_prof_type == 99:
        if verbose >= 1:
            print('Reading in RHUBC-2 AER GVRP-retrieval radiosonde data to constrain the WV profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + sorted(glob.glob(wv_prof_path + '/' + '*sonde*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >=1:
                print('No AER GVRP radiosondes found in this directory for this date')
            external['nQprof'] = 0
        else:
            maxht = int(wv_prof_maxht + 0.1)
            if maxht < wv_prof_maxht:
                maxht = maxht + 1
            zzq = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]
            
            external['nQprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()
                
                z = (z-z[0])/1000.
                
                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103))[0]
                
                if len(foo) < 2:
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                if np.nanmax(z) < maxht:
                    continue # The sonde must be above this altitude to be used here
                    
                w = Calcs_Conversions.rh2w(t, u/100., p)
                we = np.ones(len(w))*0.00001             # Assuming no error in GVRP retrieval
                    
                qunit = 'g/kg'
                qtype = 'AER profile for RHUBC-2'
                    
                # Append the data to the growing structure
                    
                if external['nQprof'] <= 0:
                    qsecs = bt+to[0]
                    wv = np.interp(zzq,z,w)
                    swv = np.interp(zzq,z,we)
                else:
                    qsecs = np.append(qsecs,bt+to[0])
                    wv = np.vstack((wv, np.interp(zzq,z,w)))
                    swv = np.vstack((swv, np.interp(zzq,z,we)))
                    
                external['nQprof'] += 1
                    
            if external['nQprof'] > 0:
                wv = wv.T
                swv = swv.T
        
    # An undefined external WV data source was specified...
    else:
        print('Error in read_external_profile_data: Undefined external water vapor profile source specified')
        return external
    
    # Read the external temperature data next
    # No external temperature profile source specified.....
    tunit = ' '
    if temp_prof_type == 0:
        a = 0          # Do nothing -- read nothing -- make no noise at all
        ttype = 'none'
        external['nTprof'] = 0
        
    # Read in the ARM radiosondes as a constraint on the temperature profile over some range
    
    elif temp_prof_type == 1:
        if verbose >= 1:
            print('Reading in ARM radiosonde data to constrain the temp profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + sorted(glob.glob(temp_prof_path + '/' + '*sonde*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM radiosondes found in this directory for this date')
            external['nTprof'] = 0.
            
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' ARM radiosonde files')
            maxht = int(temp_prof_maxht+0.1)
            if (maxht < temp_prof_maxht):
                maxht += 1
            zzt = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]
            
            external['nTprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()
                
                z = (z-z[0])/1000.
                
                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103))[0]
                
                if len(foo) < 2:
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                if np.nanmax(z) < maxht:
                    continue # The sonde must be above this altitude to be used here
                    
                sigma_t = 0.5     # The assumed uncertainty of the radiosonde temperature measurement
                
                tunit = 'C'
                ttype = 'ARM radiosonde'
                
                # Append the data to the growing structure
                if external['nTprof'] <= 0:
                    tsecs = bt+to[0]
                    temp = np.interp(zzt,z,t)
                    stemp = np.ones(len(zzt))*sigma_t
                else:
                    tsecs = np.append(tsecs,bt+to[0])
                    temp = np.vstack((temp, np.interp(zzt,z,t)))
                    stemp = np.vstack((stemp, np.ones(len(zzt))*sigma_t))
                external['nTprof'] += 1
                 
            if external['nTprof'] > 0:
                temp = temp.T
                stemp = stemp.T
    # Read in the ARM Raman lidar data (rlproftemp)
    
    elif temp_prof_type == 2:
        if verbose >= 1:
            print('Reading in ARM Raman lidar (rlproftemp) data to constrain the temp profile')
        
        ttype = 'ARM Raman lidar (rlproftemp)'
        tunit = 'C'
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(temp_prof_path + '/' + '*rlproftemp*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM RLID TEMP found in this directory for this date')
            external['nTprof'] = 0      
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' ARM RLID TEMP data files')
            
            nprof = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                
                #There are two types of RLPROF_TEMP data.
                #This handles the rlprofmr1news dataset
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'rot_raman_temperature')[0]) > 0:
                    ttype = 'ARM Raman lidar (rlprofmr1news)'
                    htx = fid.variables['height'][:]
                    tempx = fid.variables['rot_raman_temperature'][:]
                    stempx = fid.variables['rot_raman_temperature_error'][:]
                    
                    # Only keep data above 70 m to ensure the 60-m tower isn't included
                    
                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('This should not happen when processing the RLID WV data')
                        
                    htx = htx[foo]
                    tempx = tempx[foo,:]
                    stempx = stempx[foo,:]
                    
                else:
                        
                    # This handles the rlprofmr2news dataset
                        
                    ttype = 'ARM Raman lidar (rlprofmr2news)'
                    htx = fid.variables['height'][:]
                    tempx = fid.variables['temperature'][:]
                    stempx = fid.variables['temperature_error'][:]
                    
                    fid.close()
                    
                    # Only keep data above 70 m to ensure the 60-m tower isn't included
                    
                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('This should not happen when processing the RLID WV data')
                        
                    htx = htx[foo]
                    tempx = tempx[foo,:]
                    stempx = stempx[foo,:]
                    
                
                # Now append the data to the growing structure
                
                if nprof == 0:
                    tsecs = bt+to
                    temp = np.copy(tempx)
                    stemp = np.copy(stempx)
                else:
                    tsecs = np.append(tsecs, bt+to)
                    temp = np.append(temp,tempx, axis = 1)
                    swv = np.append(stemp,stempx, axis = 1)
                nprof = len(tsecs)
            
            temp = temp - 273.16
            external['nTprof'] = len(tsecs)
            zzt = np.copy(htx)
        
    # Read in the numerical weather model soundings (Greg Blumberg's format)
    elif temp_prof_type == 4:
        if verbose >= 1:
            print('Reading in NWP model output to constrain the temperature profile')
            
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(temp_prof_path + '/' + '*model*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No NWP model output data found in this directory for this date')
            external['nTprof'] = 0
        else:
            if verbose >= 2:
                print('Reading ' + str(len(files)) + ' NWP output temp files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                zzt = fid.variables['height'][:]
                ttx = fid.variables['temperature'][:]
                ssx = fid.variables['sigma_temperature'][:]
                model_type = fid.model
                model_lat = np.float(fid.gridpoint_lat)
                model_lon = np.float(fid.gridpoint_lon)
                fid.close()
                if i == 0:
                    tsecs = bt+to
                    temp = np.copy(ttx)
                    stemp = np.copy(ssx)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.vstack((temp, ttx))
                    stemp = np.vstack((stemp,ssx))
                    
            temp = temp.T
            stemp = stemp.T
            external['nTprof'] = len(tsecs)
                
            # Perform a quick check to make sure all profiles exist. I am
            # assuming that (since this is a model) I only have to look at the 
            # first level. If there are no valid profiles (ie, all of them are
            # NaN, then reset the counter to zero
            
            foo = np.where(((~np.isnan(temp[0,:])) & (np.abs(temp[0,:] < 500))))[0]
            if len(foo) > 0:
                temp = temp[:,foo]
                stemp = stemp[:,foo]
                tsecs = tsecs[foo]
                external['nTprof'] = len(tsecs)
            else:
                external['nQprof'] = 0
            
        tunit = 'C'
        ttype = 'NWP model output from ' + model_type + ' at ' + str(model_lat) + ' degN, ' + str(model_lon) + ' degE'
    
    # Read in the RHUBC-2 radiosonde data from AER's files
    elif temp_prof_type == 99:
        if verbose >= 1:
            print('Reading in RHUBC-2 AER radiosonde data to constrain the temperature profile')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + sorted(glob.glob(temp_prof_path + '/' + '*sonde*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM radiosondes found in this directory for this date')
            external['nTprof'] = 0
        else:
            maxht = int(temp_prof_maxht + 0.1)
            if maxht < temp_prof_maxht:
                maxht = maxht + 1
            zzt = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]
            
            external['nTprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()
                
                z = (z-z[0])/1000.
                
                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103))[0]
        
                if len(foo) < 2:
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                if np.nanmax(z) < maxht:
                    continue # The sonde must be above this altitude to be used here
                    
                # The assumed uncertainty of the radiosonde temperature measurement that
                # Dave hard coded
                
                sigma_t = np.interp(z,[0,2.5,3.5,20],[1.5,1.5,0.5,0.5])
                tunit = 'C'
                ttype = 'AER profile for RHUBC-2'
                
                # Append the data to the growing structure
                
                if external['nTprof'] <= 0:
                    tsecs = bt+to[0]
                    temp = np.interp(zzt,z,t)
                    stemp = np.interp(zzt,z,sigma_t)
                else:
                    tsecs = np.append(tsecs,bt+to[0])
                    temp = np.vstack((temp, np.interp(zzt,z,t)))
                    stemp = np.vstack((stemp, np.interp(zzt,z,sigma_t)))
                external['nTprof'] += len(tsecs)
            
            if external['nTprof'] > 0:
                temp = temp.T
                stemp = stemp.T
    
    else:
        print('Error in read_external_profile_data: Undefined external temperature source specified')
        return external
    
    #If there were neither input external temperature or humidity data then return gracefully
    
     
    if ((external['nQprof'] == 0) & (external['nTprof'] == 0)):
        external['success'] = 1
        return external
    elif ((external['nQprof'] < 0) | (external['nTprof'] < 0)):
        if verbose >= 2:
            print('Error associated with read_external_profile_data -- inconsistent processing somehow')
        return external
    
    # Interpolate the data to the time/height grid
    # Put the external data on the same vertical grid as the AERIoe retrievals.
    # There are many ways this could be done, but since sometimes the AERI has
    # better vertical resolution than the external source (near the surface) or
    # vice versa (aloft), we are just going to do something simple and linearly
    # interpolate and then put the data on the same temporal grid as the AERI retrievels.
    # Again, there are multiple ways to do this. I will interpolate linearly with time, but then
    # flag the samples that are within the temporal resolution window of the AERIoe
    # sample time, so tht this information can be passed to the output file.
    
    timeflag = np.zeros(len(secs))  # this captures the time flags
    if tres == 0:
        timeres = 30         # since the AERI's rapid-sample data is order of 30 seconds
    else:
        timeres = tres * 60       # because the units here should be seconds
    
    # Humidity first
    if external['nQprof'] > 0:
        # First interpolate to the correct AERIoe vertical grid. But also
        # look for bad data in the raw profile. If it exists, find the level
        # just below this in the new interpolated data and replace all points
        # above it with missing
        
         
        tmp_water = np.zeros((len(ht),len(qsecs)))
        tmp_swater = np.zeros((len(ht), len(qsecs)))
        new_water = np.zeros((len(ht), len(secs)))
        new_swater = np.zeros((len(ht), len(secs)))

        for i in range(external['nQprof']):
            tmp_water[:,i] = np.interp(ht,zzq,wv[:,i])
            tmp_swater[:,i] = np.interp(ht,zzq,swv[:,i])
            foo = np.where((wv[:,i] < 0) & (zzq >= wv_prof_minht))[0]
            if len(foo) > 0:
                bar = np.where(ht > zzq[foo[0]])[0]
                if len(bar) == 0:
                    tmp_water[:,i] = np.nan
                else:
                    tmp_water[bar[0]-1:len(ht),i] = np.nan
        
        # But set the data below or above the min/max values to a missing value
        foo = np.where((ht < wv_prof_minht) | (wv_prof_maxht < ht))[0]
        if len(foo) > 0:
            tmp_water[foo,:] = np.nan
            tmp_swater[foo,:] = np.nan
        

        # Now interpolate to the AERIoe temporal grid
        if external['nQprof'] == 1:
            # If there is only a single sample, then set all of the profiles to this
            # same profile
            for j in range(len(secs)):
                new_water[:,j] = np.copy(tmp_water)
                new_swater[:,j] = np.copy(tmp_swater)
        else:
            for j in range(len(ht)):
                new_water[j,:] = np.interp(secs,qsecs,tmp_water[j,:])
                new_swater[j,:] = np.interp(secs,qsecs,tmp_swater[j,:])
        
        # Set the interpolated data before and after the end external times to missing
        # but we will allow the interpolated data to be used over some delta_time
        # around the end times
        
        foo = np.where(((np.nanmin(qsecs)-wv_time_delta*3600) > secs) |
                       ((np.nanmax(qsecs)+wv_time_delta*3600) < secs))[0]
                       
        if len(foo) > 0:
            new_water[:,foo] = np.nan
            new_swater[:,foo] = np.nan
        
        # Now loop over the samples, and see if there is an external sample
        # that is within tres of the AERIoe time. If so, then flag this
        
        for i in range(len(secs)):
            foo = np.where((secs[i]-timeres <= qsecs) & (secs[i]+2*timeres >= qsecs))[0]
            if len(foo) > 0:
                timeflag[i] = 2                   # Use 2 for water vapor flag ("1" for temp)
        
        # Replace any NaN data with missing values
        new_water[np.isnan(new_water)] = -999
        new_swater[np.isnan(new_swater)] = -999
         
        # Now temperature....
    if external['nTprof'] > 0:
        # First interpolate to the correct AERIoe vertical grid. But also
        # look for bad data in the raw profile. If it exists, find the level
        # just below this in the new interpolated data and replace all points
        # above it with missing
            
        tmp_temp = np.zeros((len(ht),len(tsecs)))
        tmp_stemp = np.zeros((len(ht), len(tsecs)))
        new_temp = np.zeros((len(ht), len(secs)))
        new_stemp = np.zeros((len(ht), len(secs)))
         
        for i in range(external['nTprof']):
            tmp_temp[:,i] = np.interp(ht,zzt,temp[:,i])
            tmp_stemp[:,i] = np.interp(ht,zzt,stemp[:,i])
            foo = np.where((temp[:,i] < -900) & (zzt >= temp_prof_minht))[0]
            if len(foo) > 0:
                bar = np.where(ht > zzt[foo[0]])[0]
                if len(bar) == 0:
                    tmp_temp[:,i] = np.nan
                else:
                    tmp_temp[bar[0]-1:len(ht),i] = np.nan
        
        # But set the data below or above the min/max values to a missing value
        foo = np.where((ht < temp_prof_minht) | (temp_prof_maxht < ht))[0]
        if len(foo) > 0:
            tmp_temp[foo,:] = np.nan
            tmp_stemp[foo,:] = np.nan
    
        # Now interpolate to the AERIoe temporal grid
        if external['nTprof'] == 1:
            # If there is only a single sample, then set all of the profiles to this
            # same profile
            for j in range(len(secs)):
                new_temp[:,j] = np.copy(tmp_temp)
                new_stemp[:,j] = np.copy(tmp_stemp)
        else:
            for j in range(len(ht)):
                new_temp[j,:] = np.interp(secs,tsecs,tmp_temp[j,:])
                new_stemp[j,:] = np.interp(secs,tsecs,tmp_stemp[j,:])

        # Set the interpolated data before and after the end external times to missing
        # but we will allow the interpolated data to be used over some delta_time
        # around the end times
        
        foo = np.where(((np.nanmin(tsecs)-temp_time_delta*3600) > secs) |
                    ((np.nanmax(tsecs)+temp_time_delta*3600) < secs))[0]
                   
        if len(foo) > 0:
            new_temp[:,foo] = np.nan
            new_stemp[:,foo] = np.nan
        
        # Now loop over the samples, and see if there is an external sample
        # that is within tres of the AERIoe time. If so, then flag this
        
        for i in range(len(secs)):
            foo = np.where((secs[i]-timeres <= tsecs) & (secs[i]+2*timeres >= tsecs))[0]
            if len(foo) > 0:
                timeflag[i] = 1                   # Use 1 for temperature flag ("2" for WV)
    
        # Replace any NaN data with missing values
        new_temp[np.isnan(new_temp)] = -999
        new_stemp[np.isnan(new_stemp)] = -999
    
    # Apply the multiplier to the WV noise profile, if specified and if
    # the current value is not a missing value
    if external['nQprof'] > 0:
        wv_noise_multiplier = np.interp(ht, wv_noise_multiplier_hts, wv_noise_multiplier_val)
        foo = np.where(wv_noise_multiplier < 1)[0]
        if len(foo) > 0:
            print('Error in read_external_profile_data: wv_noise_multiplier_val must be >= 1')
            return external
        if verbose >= 2:
            print('Applying external_wv_noise_multiplier')
        for j in range(len(secs)):
            feh = np.where(new_swater[:,j] > -900)[0]
            if len(feh) > 0:
                new_swater[feh,j] *= wv_noise_multiplier[feh]
        
    # Apply the additive factor to the temperature noise profile, if specified
    # the current value is not a missing value
    
    if external['nTprof'] > 0:
        temp_noise_adder = np.interp(ht, temp_noise_adder_hts, temp_noise_adder_val)
        foo = np.where(temp_noise_adder < 0)[0]
        if len(foo) > 0:
            print('Error in read_external_profile_data: temp_noise_adder should not be negative')
            return external
        if verbose >= 2:
            print('Applying externl_temp_noise_adder')
        for j in range(len(secs)):
            feh = np.where(new_stemp[:,j] > -900)[0]
            if len(feh) > 0:
                new_stemp[feh,j] += temp_noise_adder[feh]
    
    # Build the output structure properly
    if ((external['nQprof'] > 0) & (external['nTprof'] > 0)):
        external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'wvmultiplier':wvmultiplier, 'wv':new_water, 'sig_wv':new_swater, 'temp':new_temp, 'sig_temp':new_stemp,
                    'wvminht':wv_prof_minht, 'wvmaxht':wv_prof_maxht, 'tempminht':temp_prof_minht, 'tempmaxht':temp_prof_maxht,
                    'timeflag':timeflag, 'wv_type':wv_prof_type, 'temp_type':temp_prof_type, 'tunit':tunit, 'qunit':qunit,
                    'ttype':ttype, 'qtype':qtype})
    
    elif external['nQprof'] > 0:
        external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'wvmultiplier':wvmultiplier, 'wv':new_water, 'sig_wv':new_swater,
                    'wvminht':wv_prof_minht, 'wvmaxht':wv_prof_maxht,
                    'timeflag':timeflag, 'wv_type':wv_prof_type, 'qunit':qunit,
                    'qtype':qtype})
    
    elif external['nTprof'] > 0:
         external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'temp':new_temp, 'sig_temp':new_stemp,
                    'tempminht':temp_prof_minht, 'tempmaxht':temp_prof_maxht,
                    'timeflag':timeflag, 'temp_type':temp_prof_type, 'tunit':tunit,
                    'ttype':ttype})
    
    else:
         external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'],
                    'wv_type':wv_prof_type, 'temp_type':temp_prof_type, 'tunit':tunit, 'qunit':qunit,
                    'ttype':ttype, 'qtype':qtype})
    
    return external

################################################################################
# This function reads in time-series data that will be used to help constrain the
# retrieval (e.g., surface met data)
################################################################################

def read_external_timeseries(date, secs, tres, avg_instant, sfc_temp_type,
            sfc_wv_type, sfc_path, sfc_temp_npts, sfc_wv_npts, sfc_temp_rep_error, sfc_wv_mult_error,
            sfc_wv_rep_error, sfc_time_delta, sfc_relative_height, co2_sfc_type,
            co2_sfc_npts, co2_sfc_rep_error, co2_sfc_path, co2_sfc_relative_height,
            co2_sfc_time_delta, use_ext_psfc, dostop, verbose):
            
    external = {'success':0, 'nTsfc':-1, 'nQsfc':-1, 'nCO2sfc':-1}
    ttype = 'None'
    qtype = 'None'
    co2type = 'None'
    
    # Some quick QC of the input entries in the VIP file
    estring = ' '
    maxpts = 1000                         # Maximum number of surface replication points allowed
    if sfc_temp_npts > 0:
        if ((sfc_temp_npts < 1) | (maxpts < sfc_temp_npts)):
            estring = 'VIP input error: when sfc_temp_type > 0, then 1 <= ext_sfc_temp_npts < ' + str(maxpts)
    if sfc_wv_type > 0:
        if ((sfc_wv_npts < 1) | (maxpts < sfc_wv_npts)):
            estring = 'VIP input error; when sfc_wv_type > 0, then 1 <= ext_sfc_wv_npts < ' + str(maxpts)
    if co2_sfc_type > 0:
        if ((co2_sfc_npts < 1) | (maxpts < co2_sfc_npts)):
            estring = 'VIP input error; when co2_sfc_npts > 0, then 1 <= co2_sfc_npts < ' + str(maxpts)
    if use_ext_psfc > 0:
        if ((sfc_temp_type == 0) & (sfc_wv_type == 0)):
            # TODO - Make this not rely on having another sfc observation
            estring = 'VIP input error; if use_ext_psfc > 0, either ext_sfc_temp_type or ext_sfc_wv_type must be > 0'

    if estring != ' ':
        if verbose >= 1:
            print(estring)
        return external
    
    # Read the external surface met temperature data first
    # No external surface temperature source specified
    
    tunit = ' '
    
    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]
    
    cdf = ['nc','cdf']
    
    if sfc_temp_type == 0:
        a = 0                 # Do nothing -- read nothing -- make no noise at all
        ttype = 'none'
        external['nTsfc'] = 0
        
    # Read in the ARM met temperature data
    elif (sfc_temp_type == 1):
        if verbose >= 1:
            print('Reading in ARM met temperature data')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(sfc_path + '/' + '*met*' + dates[i] + '*.' + cdf[j]))
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM met found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][:]
                to = fid.variables['time_offset'][:]
                p = fid.variables['atmos_pressure'][:]        # kPa
                t = fid.variables['temp_mean'][:]             # degC
                u = fid.variables['rh_mean'][:]               # %RH
                fid.close()
                p *= 10.                 # Convert kPa to hPa
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                tunit = 'C'
                ttype = 'ARM met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5
                if external['nTsfc'] <= 0:
                    tsecs = bt+to
                    temp = np.copy(t)
                    stemp = np.ones(len(t))*sigma_t
                    press = np.copy(p)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.append(temp,t)
                    stemp = np.append(stemp,np.ones(len(t))*sigma_t)
                    press = np.append(press, p)
                external['nTsfc'] = len(tsecs)
        
        # Read in the NCAR ISFS data
    
    elif sfc_temp_type == 2:
        if verbose >= 1:
            print('Reading in NCAR ISFS met temperature data')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(sfc_path + '/' + '*isfs*' + dates[i] + '*.' + cdf[j]))
        
        # Some folks are creating surface met data with the same data format
        # as the EOL ISFS dataset, but using "met" as the rootname. So if there
        # are no ISFS files found,then try looking for met instead before aborting.
        
        if len(files) == 0:
           for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + (glob.glob(sfc_path + '/' + '*met*' + dates[i] + '*.' + cdf[j]))
                
        if len(files) == 0:
            if verbose >= 1:
                print('No NCAR ISFS met found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][:]
                if len(np.where(np.array(list(fid.variables.keys())) == 'time')[0]) > 0:
                    to = fid.variables['time'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
                    to = fid.variables['time_offset'][:]
                else:
                    print('Error: Unable to find the time field in the ISFS data file')
                    fid.close()
                    return external
                p = fid.variables['pres'][:]            # hPa
                t = fid.variables['tdry'][:]            # degC
                u = fid.variables['rh'][:]              # %RH
                
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                tunit = 'C'
                ttype = 'NCAR ISFS met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5
                if external['nTsfc'] <= 0:
                    tsecs = bt+to
                    temp = np.copy(t)
                    stemp = np.ones(len(t))*sigma_t
                    press = np.copy(p)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.append(temp,t)
                    stemp = np.append(stemp,np.ones(len(t))*sigma_t)
                    press = np.append(press, p)
                external['nTsfc'] = len(tsecs)
        
        # Read in the microwave radiometer met data
        
    elif sfc_temp_type == 3:
        if verbose >= 1:
            print('Reading in MWR met temperature data')
        
        files = []        
        names = ['mwr','met']
        for i in range(len(dates)):
            for j in range(len(cdf)):
                for k in range(len(names)):
                    files = files + (glob.glob(sfc_path + '/' + '*' + names[k] + '*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No MWR met found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][:]
                to = fid.variables['time_offset'][:]
                
                #This field could be sfc_pres or p_sfc
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_pres')[0]) > 0:
                    p = fid.variables['sfc_pres'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'p_sfc')[0]) > 0:
                    p = fid.variables['p_sfc'][:]
                else:
                    p = np.ones(len(to))*-999.
                
                # This field could be sfc_temp or t_sfc. I also need to check the 
                # units to make sure that is handled correctly
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_temp')[0]) > 0:
                    t = fid.variables['sfc_temp'][:]
                    ttunit = fid.variables['sfc_temp'].units
                    if ((ttunit == 'K') | (ttunit == 'k') | (ttunit == 'Kelvin')):
                        t -= 273.15
                elif len(np.where(np.array(list(fid.variables.keys())) == 't_sfc')[0]) > 0:
                    t = fid.variables['t_sfc'][:]
                    ttunit = fid.variables['t_sfc'].units
                    if ((ttunit == 'K') | (ttunit == 'k') | (ttunit == 'Kelvin')):
                        t -= 273.15
                else:
                    t = np.ones(len(to))*-999.
                    
                # This field could be sfc_rh or rh_sfc
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_rh')[0]) > 0:
                    u = fid.variables['sfc_rh'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'rh_sfc')[0]) > 0:
                    u = fid.variables['rh_sfc'][:]
                else:
                    u = np.ones(len(to))*-999.
                fid.close()
                
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                tunit = 'C'
                ttype = 'Microwave radiometer met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5
                if external['nTsfc'] <= 0:
                    tsecs = bt+to
                    temp = np.copy(t)
                    stemp = np.ones(len(t))*sigma_t
                    press = np.copy(p)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.append(temp,t)
                    stemp = np.append(stemp,np.ones(len(t))*sigma_t)
                    press = np.append(press, p)
                external['nTsfc'] = len(tsecs)
        
        # An undefined external surface met temperature source was specified...
    
    else:
        print('Error in read_external_tseries: Undefined external met temperature source specified')
        return external
    
    # Read the external surface met water vapor data next
    # No external surface water vapor source specified...
    
    qunit = ' '
    if sfc_wv_type == 0:
        a = 0                # Do nothing -- read nothing -- make no noise at all
        qtype = 'none'
        external['nQsfc'] = 0
        
    # Read in the ARM met water vapor data
    
    elif sfc_wv_type == 1:
        if verbose >= 1:
            print('Reading in ARM met water vapor data')
        
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(sfc_path + '/' + '*met*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM met found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][:]
                to = fid.variables['time_offset'][:]
                p = fid.variables['atmos_pressure'][:]        # kPa
                t = fid.variables['temp_mean'][:]             # degC
                u = fid.variables['rh_mean'][:]               # %RH
                fid.close()
                p *= 10.                 # Convert kPa to hPa
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo].squeeze()
                p = p[foo].squeeze()
                t = t[foo].squeeze()
                qunit = 'g/kg'
                qtype = 'ARM met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5     # degC
                sigma_u = 3.0     # %RH
                w0 = Calcs_Conversions.rh2w(t, u/100., p)
                w1 = Calcs_Conversions.rh2w(t+sigma_t, u/100., p)
                w2 = Calcs_Conversions.rh2w(t-sigma_t, u/100., p)
                u_plus = (u+sigma_u)/100.
                u_minus = (u+sigma_u)/100.
                u_plus[u_plus > 1] == 1
                u_minus[u_minus < 0] == 0
                w3 = Calcs_Conversions.rh2w(t, u_plus, p)
                w4 = Calcs_Conversions.rh2w(t, u_minus, p)
                
                # Sum of squared errors, but take two-side average for T and RH uncerts
                sigma_w = np.sqrt( ((w1-w0)**2 + (w2-w0)**2)/2. + ((w3-w0)**2 + (w4-w0)**2)/2. )
                if external['nQsfc'] <= 0:
                    qsecs = bt+to
                    wv = np.copy(w0)
                    swv = np.copy(sigma_w)
                    press = np.copy(p)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,w0)
                    swv = np.append(swv,sigma_w)
                    press = np.append(press, p)
                external['nQsfc'] = len(qsecs)
        
        # Read in the NCAR ISFS data
        
    elif sfc_wv_type == 2:
        if verbose >= 1:
            print('Reading in NCAR ISFS met water vapor data')
            
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(sfc_path + '/' + '*isfs*' + dates[i] + '*.' + cdf[j]))
        
        # Some folks are creating surface met data with the same data format
        # as the EOL ISFS dataset, but using "met" as the rootname. So if there
        # are no ISFS files found,then try looking for met instead before aborting.
        
        if len(files) == 0:
           for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + (glob.glob(sfc_path + '/' + '*met*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No NCAR ISFS met found in this directory for this date')
        
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][:]
                if len(np.where(np.array(list(fid.variables.keys())) == 'time')[0]) > 0:
                    to = fid.variables['time'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
                    to = fid.variables['time_offset'][:]
                else:
                    print('Error: Unable to find the time field in the ISFS data file')
                    fid.close()
                    return external
                p = fid.variables['pres'][:]            # hPa
                t = fid.variables['tdry'][:]            # degC
                u = fid.variables['rh'][:]              # %RH
                
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                qunit = 'g/kg'
                qtype = 'NCAR ISFS met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5      # degC
                sigma_u = 3.0      # %RH
                w0 = Calcs_Conversions.rh2w(t, u/100., p)
                w1 = Calcs_Conversions.rh2w(t+sigma_t, u/100., p)
                w2 = Calcs_Conversions.rh2w(t-sigma_t, u/100., p)
                u_plus = (u+sigma_u)/100.
                u_minus = (u+sigma_u)/100.
                u_plus[u_plus > 1] == 1
                u_minus[u_minus < 0] == 0
                w3 = Calcs_Conversions.rh2w(t, u_plus, p)
                w4 = Calcs_Conversions.rh2w(t, u_minus, p)
                
                # Sum of squared errors, but take two-side average for T and RH uncerts
                sigma_w = np.sqrt( ((w1-w0)**2 + (w2-w0)**2)/2. + ((w3-w0)**2 + (w4-w0)**2)/2. )
                if external['nQsfc'] <= 0:
                    qsecs = bt+to
                    wv = np.copy(w0)
                    swv = np.copy(sigma_w)
                    press = np.copy(p)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,w0)
                    swv = np.append(swv,sigma_w)
                    press = np.append(press, p)
                external['nQsfc'] = len(qsecs)
        
        # Read in the MWR met data
    elif sfc_wv_type == 3:
        if verbose >= 1:
            print('Reading in MWR met water vapor data')
            
        names = ['mwr','met']
        for i in range(len(dates)):
            for j in range(len(cdf)):
                for k in range(len(names)):
                    files = files + (glob.glob(sfc_path + '/' + '*' + names[k] + '*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No MWR met found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][:]
                to = fid.variables['time_offset'][:]
                
                #This field could be sfc_pres or p_sfc
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_pres')[0]) > 0:
                    p = fid.variables['sfc_pres'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'p_sfc')[0]) > 0:
                    p = fid.variables['p_sfc'][:]
                else:
                    p = np.ones(len(to))*-999.
                
                # This field could be sfc_temp or t_sfc. I also need to check the 
                # units to make sure that is handled correctly
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_temp')[0]) > 0:
                    t = fid.variables['sfc_temp'][:]
                    ttunit = fid.variables['sfc_temp'].units
                    if ((ttunit == 'K') | (ttunit == 'k') | (ttunit == 'Kelvin')):
                        t -= 273.15
                elif len(np.where(np.array(list(fid.variables.keys())) == 't_sfc')[0]) > 0:
                    t = fid.variables['t_sfc'][:]
                    ttunit = fid.variables['t_sfc'].units
                    if ((ttunit == 'K') | (ttunit == 'k') | (ttunit == 'Kelvin')):
                        t -= 273.15
                else:
                    t = np.ones(len(to))*-999.
                    
                # This field could be sfc_rh or rh_sfc
                
                if len(np.where(np.array(list(fid.variables.keys())) == 'sfc_rh')[0]) > 0:
                    u = fid.variables['sfc_rh'][:]
                elif len(np.where(np.array(list(fid.variables.keys())) == 'rh_sfc')[0]) > 0:
                    u = fid.variables['rh_sfc'][:]
                else:
                    u = np.ones(len(to))*-999.
                fid.close()
                
                foo = np.where((p > 0) & (p < 1050) & (t < 60) & (u >= 0) & (u < 103))[0]
                if len(foo) < 2:
                    continue
                to = to[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]
                qunit = 'g/kg'
                qtype = 'Microwave radiometer met station'
                
                # Append the data to the growing structure
                sigma_t = 0.5      # degC
                sigma_u = 3.0      # %RH
                w0 = Calcs_Conversions.rh2w(t, u/100., p)
                w1 = Calcs_Conversions.rh2w(t+sigma_t, u/100., p)
                w2 = Calcs_Conversions.rh2w(t-sigma_t, u/100., p)
                u_plus = (u+sigma_u)/100.
                u_minus = (u+sigma_u)/100.
                u_plus[u_plus > 1] == 1
                u_minus[u_minus < 0] == 0
                w3 = Calcs_Conversions.rh2w(t, u_plus, p)
                w4 = Calcs_Conversions.rh2w(t, u_minus, p)
                
                # Sum of squared errors, but take two-side average for T and RH uncerts
                sigma_w = np.sqrt( ((w1-w0)**2 + (w2-w0)**2)/2. + ((w3-w0)**2 + (w4-w0)**2)/2. )
                if external['nQsfc'] <= 0:
                    qsecs = bt+to
                    wv = np.copy(w0)
                    swv = np.copy(sigma_w)
                    press = np.copy(p)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,w0)
                    swv = np.append(swv,sigma_w)
                    press = np.append(press, p)
                external['nQsfc'] = len(qsecs)
        
        # An undefined external surface met water vapor source was specified
    else:
        print('Error in read_external_tseries: Undefined external met water vapor source')
        return external
    
    # Add on the representativeness errors that were specified
    if ((external['nTsfc'] > 0) & (sfc_temp_rep_error > 0)):
        stemp += sfc_temp_rep_error
    if ((external['nQsfc'] > 0) & (sfc_wv_mult_error >= 0)):
        swv *= sfc_wv_mult_error
    if ((external['nQsfc'] > 0) & (sfc_wv_rep_error > 0)):
        swv += sfc_wv_rep_error
    
    # Now I need to bin/interpolate the data appropriately
    if external['nTsfc'] > 0:
        # Compute the median time interval between Tsfc measurements [minutes]
        tdel = np.nanmedian(tsecs[1:len(tsecs)] - tsecs[0:len(tsecs)-1]) / 60.
        
        # If the median time interval is much smaller than tavg, then we will
        # bin up the data. Otherwise, we will just interpolate linearly
        
        if (tdel*4 < tres):
            #Bin the data
            tt0 = np.zeros(len(secs))
            st0 = np.zeros(len(secs))
            p0 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-tres*60/2. <= tsecs) & (tsecs <= secs[i] + tres*60/2.))[0]
                if len(foo) > 0:
                    tt0[i] = np.nanmean(temp[foo])
                    st0[i] = np.nanmean(stemp[foo])
                    p0[i] = np.nanmean(press[foo])
                else:
                    tt0[i] = -999.
                    st0[i] = -999.
                    p0[i] = -999.
        else:
            tt0 = np.interp(secs,tsecs,temp)
            st0 = np.interp(secs,tsecs,stemp)
            p0 = np.interp(secs,tsecs,press)
            foo = np.where(secs < tsecs[0]-sfc_time_delta*3600)[0]
            if len(foo) > 0:
                tt0[foo] = -999.
                st0[foo] = -999.
                p0[foo] = -999.

            # Make sure we did not interpolate out of bounds here.
            foo = np.where((tsecs[0]-sfc_time_delta*3600 <= secs) & (secs < tsecs[0]))[0]
            if len(foo) > 0:
                tt0[foo] = temp[0]
                st0[foo] = stemp[0]
                p0[foo] = press[0]
            n = len(tsecs) - 1
            foo = np.where(tsecs[n]+sfc_time_delta*3600 < secs)[0]
            if len(foo) > 0:
                tt0[foo] = -999.
                st0[foo] = -999.
                p0[foo] = -999.
            foo = np.where((tsecs[n] < secs) & (secs <= tsecs[n]+sfc_time_delta*3600))[0]
            if len(foo) > 0:
                tt0[foo] = temp[0]
                st0[foo] = stemp[0]
                p0[foo] = press[0]
    else:
        tt0 = -999.
        st0 = -999.
        p0 = -999.
    
    if external['nQsfc'] > 0:
        # Compute the median time interval between Tsfc measurements [minutes]
        tdel = np.nanmedian(qsecs[1:len(qsecs)] - qsecs[0:len(qsecs)-1]) / 60.
        
        # If the median time interval is much smaller than tavg, then we will
        # bin up the data. Otherwise, we will just interpolate linearly
        
        if (tdel*4 < tres):
            #Bin the data
            qq0 = np.zeros(len(secs))
            sq0 = np.zeros(len(secs))
            p0 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-tres*60/2. <= qsecs) & (qsecs <= secs[i] + tres*60/2.))[0]
                if len(foo) > 0:
                    qq0[i] = np.nanmean(wv[foo])
                    sq0[i] = np.nanmean(swv[foo])
                    p0[i] = np.nanmean(press[foo])
                else:
                    qq0[i] = -999.
                    sq0[i] = -999.
                    p0[i] = -999.
        else:
            qq0 = np.interp(secs,qsecs,wv)
            sq0 = np.interp(secs,qsecs,swv)
            p0 = np.interp(secs,qsecs,press)
            foo = np.where(secs < qsecs[0]-sfc_time_delta*3600)[0]
            if len(foo) > 0:
                qq0[foo] = -999.
                sq0[foo] = -999.
                p0[foo] = -999.
            
            # Make sure we did not interpolate out of bounds here.
            foo = np.where((qsecs[0]-sfc_time_delta*3600 <= secs) & (secs < qsecs[0]))[0]
            if len(foo) > 0:
                qq0[foo] = wv[0]
                sq0[foo] = swv[0]
                p0[foo] = press[0]
            n = len(qsecs) - 1
            foo = np.where(qsecs[n]+sfc_time_delta*3600 < secs)[0]
            if len(foo) > 0:
                qq0[foo] = -999.
                sq0[foo] = -999.
                p0[foo] = -999.
            foo = np.where((qsecs[n] < secs) & (secs <= qsecs[n]+sfc_time_delta*3600))[0]
            if len(foo) > 0:
                qq0[foo] = wv[0]
                sq0[foo] = swv[0]
                p0[foo] = -999.
    else:
        qq0 = -999.
        sq0 = -999.
        p0 = -999.

    # This section is for the CO2 obs
    # Read in the surface in-situ CO2 data, if desired
    # No external surface CO2 source specified....
    co2unit = ' '
    if co2_sfc_type == 0:
        a = 0                  # Do nothing -- read nothing -- make no noise at all
        co2type = 'none'
        external['nCo2sfc'] = 0
    
    # Read in the surface in-situ CO2 data (assuming DDT's PGS qc1turn datastream)
    elif co2_sfc_type == 1:
        if verbose >= 1:
            print('Reading in ARM PGS qc1turn datastream')
            
        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + (glob.glob(co2_sfc_path + '/' + '*pgs*' + dates[i] + '*.' + cdf[j]))
        
        if len(files) == 0:
            if verbose >= 1:
                print('No ARM CO2 found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                bt = fid.variables['base_time'][:]
                to = fid.variables['time_offset'][:]
                xco2 = fid.variables['c02_02m'][:]                  # ppm
                xsco2 = fid.variables['sigma_co2_02m'][:]           # ppm
                fid.close()
                co2unit = 'ppm'
                co2type = 'ARM PGS qc1turn'
                
                # Append the data to the growing structure
                if external['nCo2sfc <= 0']:
                    co2secs = bt+to
                    co2 = np.copy(xco2)
                    sco2 = np.copy(xsco2)
                else:
                    co2secs = np.append(co2secs,bt+to)
                    co2 = np.append(co2,xco2)
                    sco2 = np.append(sco2,xsco2)
                external['nCO2sfc'] = len(co2secs)
        
        # An undefined surface in-situ CO2 source was specified...
    else:
        print('Error in read_external_tseries: Undefined in-situ CO2 surface obs source specified')
        return external
    
    # Strip out any missing values before the next step
    if external['nCo2sfc'] > 0:
        foo = np.where(co2 > 0)[0]
        if len(foo) > 0:
            external['nCo2sfc'] = len(foo)
            co2secs = co2secs[foo]
            co2 = co2[foo]
            sco2 = sco2[foo]
        else:
            external['nCo2sfc'] = 0
    
    # Add on the representativeness errors that were specified 
    if ((external['nCo2sfc'] > 0) & (co2_sfc_rep_error > 0)):
        sco2 += co2_sfc_rep_error
    
    # Now I need to bin/interpolate the data appropriately
    if external['nCo2sfc'] > 0:
        # Compute the median time interval between CO2 measurements [minutes]
        tdel = np.nanmedian(co2secs[1:len(co2secs)] - co2secs[0:len(co2secs)-1]) / 60.
        
        # If the median time interval is much smaller than tavg, then
        # we will bin up the data. Otherwise, we will just interpolate linearly
        
        if (tdel*4 < tres):
            # Bin the data
            cco2 = np.zeros(len(secs))
            scco2 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-tres*60/2. <= co2secs) & (co2secs <= secs[i] + tres*60/2.))[0]
                if len(foo) > 0:
                    cco2[i] = np.nanmean(co2[foo])
                    scco2[i] = np.nanmean(sco2[foo])
                else:
                    cco2[i] = -999.
                    scco2[i] = -999.
        else:
            cco2 = np.interp(secs,co2secs,co2)
            scco2 = np.interp(secs,co2secs,sco2)
            foo = np.where(secs < co2secs[0]-co2_sfc_time_delta*3600)[0]
            if len(foo) > 0:
                cco2[foo] = -999.
                scco2[foo] = -999.
            
            # Make sure we did not interpolate out of bounds here.
            foo = np.where((co2secs[0]-co2_sfc_time_delta*3600 <= secs) & (secs < co2secs[0]))[0]
            if len(foo) > 0:
                cco2[foo] = co2[0]
                scco2[foo] = sco2[0]
            n = len(co2secs) - 1
            foo = np.where(co2secs[n]+co2_sfc_time_delta*3600 < secs)[0]
            if len(foo) > 0:
                cco2[foo] = -999.
                scco2[foo] = -999.
            foo = np.where((co2secs[n] < secs) & (secs <= co2secs[n]+co2_sfc_time_delta*3600))[0]
            if len(foo) > 0:
                cco2[foo] = co2[0]
                scco2[foo] = sco2[0]
    else:
        cco2 = -999.
        scco2 = -999.
    
    # Now I need to build arrays for each suface sample, so that I can
    # replicate a point multiple times if desired. I don't want to make each
    # of the replicates the same as the original, so I will insert 1/10th of
    # the random error into the time-series...
    
    if external['nTsfc'] > 0:
        tt1 = np.zeros((sfc_temp_npts,len(secs)))
        st1 = np.zeros((sfc_temp_npts,len(secs)))
        tt1[0,:] = tt0
        st1[0,:] = st0
        for j in range(1,sfc_temp_npts):
            tt1[j,:] = tt1 = tt0 + np.random.normal(size = len(secs))*st0/10.
            st1[j,:] = np.copy(st0)
    else:
        tt1 = np.copy(tt0)
        st1 = np.copy(st0)
    
    if external['nQsfc'] > 0:
        qq1 = np.zeros((sfc_wv_npts,len(secs)))
        sq1 = np.zeros((sfc_wv_npts,len(secs)))
        qq1[0,:] = np.copy(qq0)
        sq1[0,:] = np.copy(sq0)
        for j in range(1,sfc_wv_npts):
            qq1[j,:] = qq0 + np.random.normal(size = len(secs))*sq0/10.
            sq1[j,:] = np.copy(sq0)
    else:
        qq1 = np.copy(qq0)
        sq1 = np.copy(sq0)
    
    if external['nCo2sfc'] > 0:
        cco2a = np.zeros((co2_sfc_npts,len(secs)))
        scco2a = np.zeros((co2_sfc_npts,len(secs)))
        cco2a[0,:] = np.copy(cco2)
        scco2a[0,:] = np.copy(scco2)
        for j in range(1,co2_sfc_npts):
            cco2a[j,:] = cco2 + np.random.normal(size = len(secs))*scco2/10.
            scco2a[j,:] = np.copy(scco2)
    else:
        cco2a = np.copy(cco2)
        scco2a = np.copy(scco2)
    
    # Build the output structure and return it
    
    external = {'success':1, 'sfc_relative_height':sfc_relative_height,
          'nTsfc':external['nTsfc'],'nptsT':sfc_temp_npts, 'tunit':tunit, 'ttype':ttype,
          'temp':tt1, 'stemp':st1, 'nQsfc':external['nQsfc'], 'nptsQ':sfc_wv_npts, 'qunit':qunit,
          'qtype':qtype, 'wv':qq1, 'swv':sq1, 'sfc_temp_rep_error':sfc_temp_rep_error,
          'sfc_wv_rep_error':sfc_wv_rep_error, 'sfc_wv_mult_error':sfc_wv_mult_error,
          'nCO2sfc':external['nCO2sfc'], 'co2unit':co2unit, 'nptsCO2':co2_sfc_npts,
          'co2type':co2type, 'co2':cco2a, 'sco2':scco2a, 'co2_sfc_rep_error':co2_sfc_rep_error,
          'co2_sfc_relative_height':co2_sfc_relative_height, 'psfc': p0}
    return external
