#!/bin/sh

# $Id: $

# This script makes running MIXCRA (using the TROPoe container) "easy" for operations

if [[ $# -ne 9 ]]; then 
  echo "USAGE: $0 yyyymmdd vip_file shour ehour step keep_flag verbose data_path image_id"
  echo "   where      yyyymmdd : is the date to process"
  echo "              vip_file : is the path/name of the VIP file"
  echo "                 shour : is the start hour"
  echo "                 ehour : is the  end  hour"
  echo "                  step : is the step size (integer 1 or larger)"
  echo "                  keep : is the flag to keep the LBLRTM calcs or not"
  echo "               verbose : is the verbosity level (1, 2, or 3)"
  echo "             data_path : is the location of the data tree"
  echo "   and        image_id : is the identification number or name of the Docker image to execute"
  exit
fi

echo "Running MIXCRA in operational mode, with"
echo "                            Date (yyyymmdd) : $1"
echo "                                   VIP file : $2"
echo "                                 Start hour : $3"
echo "                                   End hour : $4"
echo "                                       Step : $5"
echo "                                  Keep flag : $6"
echo "                               Verbose flag : $7"
echo "  External data directory (mapped to /data) : $8"
echo "                                 Image name : $9"


# If using Docker, then use this command
echo "Running image Docker"
docker run -it --userns=host -e "app=MIXCRA" -e "yyyymmdd=$1" -e "vfile=$2" -e "shour=$3" -e "ehour=$4" -e "step=$5" -e "keep=$6" -e "verbose=$7" -e "debug=0" -v $8:/data $9

# Else if you are using Podman, then comment the above command and uncomment this one
#echo "Running image Podman"
#podman run -it -u root --rm -e "app=MIXCRA" -e "yyyymmdd=$1" -e "vfile=$2" -e "shour=$3" -e "ehour=$4" -e "step=$5" -e "keep=$6" -e "verbose=$7" -e "debug=0" --security-opt label=disable -v $8:/data $9 
