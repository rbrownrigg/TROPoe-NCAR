-----------
DISCLAIMER
-----------
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration (NOAA), or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. NOAA has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

-----------
DESCRIPTION
-----------
TROPoe (pronounced "trope-oh-ee") is an optimal-estimation based algorithm to retrieve thermodynamic profiles from ground-based remote sensors.  It was originally designed for the AERI (hence it was previously called AERIoe), but the algorithm has been greatly extended to process other passive ground-based remote sensors like microwave radiometers (both MP3000 and HATPRO systems) and infrared spectrometers (both AERI and ASSIST systems).  It can include a wide range of additional observations such as in-situ observations at the surface or towers, profiles from water vapor lidars or RASS systems, and NWP model output.  The algorithm finds the temperature and humidity profile, as well as the corresponding cloud properties, that provides the best agreement with these observations and the input climatology (a priori dataset, which is provided externally).  The results include the retrieved profiles along with a full error characterization and information content.

TROPoe is an python version of Dave Turner's IDL-based AERIoe algorithm. The original IDL code was Release_2_9.  This code was produced so AERIoe could be run using a free programming language. The logic in TROPoe follows the original code exactly. The main driver script for TROPoe is the TROPoe.py file. This script makes calls to dozens of functions that are written in the other .py files.

The functions are separated in files by the tasks that they perform so users can find and edit them (if needed) efficiently.

VIP_Databases_function.py - Functions that read in the VIP file and other databases needed for the retrieval

Calcs_Conversions.py - Functions that perform basic meteorological calculations and variable conversions.

Data_reads.py - These functions control the reading of the all the data used in the retrieval

Jacobian_Functions - Functions that calculate the jacobians and forward operators for the retrieval

Output_Functions - These functions write the output file from the retrieval

Other_Functions - These functions don't really fit in the other categories

TROPoe depends strongly on the forward models used to map from the state space (i.e., thermodynamic profiles and cloud properties) into the observation space (i.e., microwave and/or infrared spectral radiance).  It uses the MonoRTM and LBLRTM as the forward models, respectively.  

TROPoe has been packed into a Docker container to make the maintanence and distribution of the software easier.  A bash shell script is available to easily run this container on the unix/linux command line.  A User's Guide is currently under development.

TROPoe is designed to be extremely flexible, and to be able to perform both operational-like retrievals and experimental retrievals.  The configuration of the retrieval is controlled by the VIP (variable-input parameter) file.  The VIP file is a set of key=value pairs, which are used to override the default configuration stored internally in the code.  This is described in detail by the User's Guide.

----------------------
PYTHON PACKAGES NEEDED 
----------------------
python 3.x.  Note that all of these packages are part of the Docker container already.  

os
sys
shutil
copy
numpy
scipy
netCDF4
datetime
calendar
glob
subprocess
struct
time
argparse

Note: Most of these packages are standard in python distributions


------------
INSTRUCTIONS
------------

For the retrieval to be run LBLRTM needs to be installed and the lblrun.shippert_modification needs to be added to the install. This file is included in the TROPoe distribution. If MWR data is going to be used, then MonoRTM also needs to be installed.


1. The prior needs to be computed. This can be made from compute_prior.py (not yet implimented) or an already created prior dataset can be used.

2. Edit the VIP file. This controls the input parameters for the retrieval. Read the User's Guide for info on the options available.

3. Change directory to the location of the python source code

4. Run the retrieval using the following command:

python TROPoe.py retrieval_date VIP_filename prior_filename <--shour shour> <--ehour ehour> <--verbose verbose>
  where
    retrieval_date  is in yyyymmdd format (i.e., 8-digit)
    VIP_filename    is the path/filename of the VIP file
    prior_filename  is the path/filename of the input a-priori data

Optional arguments:
    --shour     - the start hour. Default is 0.
    --ehour     - the end hour. If -1 then all AERI times are used. Default is -1.
    --verbose   - 0-3, controls the verbosity of the retrieval. 0 is very quiet, 3 is very noisy. Default is 1.

------------
REFERENCES
------------

Turner, D.D., and U. Loehnert, 2021: Ground-based temperature and humidity profiling: Combining active and passive remote sensors. Atmos. Meas. Technol., 14, 3033-3048, doi:10.5194/amt-14-3033-2021.

Turner, D.D., and W.G. Blumberg, 2019: Improvements to the AERIoe thermodynamic profile retrieval algorithm. IEEE J. Selected Topics Appl. Earth Obs. Remote Sens., 12, 1339-1354, doi:10.1109/JSTARS.2018.2874968.

Turner, D.D., and U. Loehnert, 2014: Information content and uncertainties in thermodynamic profiles and liquid cloud properties retrieved from the ground-based Atmospheric Emitted Radiance Interferometer (AERI). J. Appl. Meteor. Clim., 53, 752-771, doi:10.1175/JAMC-D-13-0126.1


------------
CONTACTS
------------

Dr. Dave Turner, NOAA Global Systems Laboratory
Email: dave.turner@noaa.gov


