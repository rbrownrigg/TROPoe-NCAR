-----------
DESCRIPTION
-----------
pyAERIoe is an python version of Dave Turner's IDL AERIoe algorithm. This code was produced so AERIoe could be run using a free programming language. The logic in pyAERIoe follows the original code exactly. The main driver script for pyAERIoe is the AERIoe.py file. This script makes calls to 79 functions that are written in the other .py files.
The functions are separated in files by the tasks that they perform so users can find and edit them (if needed) efficiently.

VIP_Databases_function.py - Functions that read in the VIP file and other databases needed for the retrieval

Calcs_Conversions.py - Functions that perform basic meteorological calculations and variable conversions.

Data_reads.py - These functions control the reading of the all the data used in the retrieval

Jacobian_Functions - Functions that calculate the jacobians and forward operators for the retrieval

Output_Functions - These functions write the output file from the retrieval

Other_Functions - These functions don't really fit in the other categories


----------------------
PYTHON PACKAGES NEEDED 
----------------------
python 2.7 (A future version of pyAERIoe will be made for python 3)

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

For the retrieval to be run LBLRTM needs to be installed and the lblrun.shippert_modification needs to be added to the install. This file is included in the pyAERIoe distribution. If MWR data is going to be used, then MonoRTM also needs to be installed.


1. The prior needs to be computed. This can be made from compute_prior.py (not yet implimented)  or an already created prior dataset can be used.

2. Edit the pyVIP file. This controls the input parameters for the retrieval. Read the AERIoe_vip_instructions.docx for info on the options available.

3. Run the retrieval using the following command:

python <path-to-AERIoe.py> <retrieval date> <pyVIP filename> <prior filename> <--shour shour> <--ehour ehour> <--verbose verbose>

Optional arguments:
--shour - the start hour. Default is 0.
--ehour - the end hour. If -1 then all AERI times are used. Default is -1.
--verbose - 0-3, controls the verbosity of the retrieval. 0 is very quiet, 3 is very noisy. Default is 1.

