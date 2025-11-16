#!/bin/sh

SENSITIVITY_MAP_PATH=/mnt/opt/groups/jpet/Software/CASTOR/scripts/Modular_JPET/merged_sens_atten/atten_sens_joseph_it1.hdr

if [[ "$#" -ne 3 ]]; then
  echo "Usage ./castor-run.sh <path to input folder> <path to output folder> <castor reconstruction output folder>"
  exit 1
fi
FOLDER=$1
OUTPUT_FOLDER=$2
RECONSTRUCTION_FOLDER=$3
FILENAME=$(basename ${RECONSTRUCTION_FOLDER})
FOLDER=$(dirname ${RECONSTRUCTION_FOLDER})
PROFILES_FOLDER=${FOLDER}/PROFILES/

EXT="*.cat.evt.root"
FILES=$(find $FOLDER -name $EXT | tr "\n" " ")
mkdir -p $OUTPUT_FOLDER

mkdir -p ${PROFILES_FOLDER}

user_params_file="userParams"
PARAMS_EXT=".json"
if [[ -e ${user_params_file}${PARAMS_EXT} || -L ${user_params_file}${PARAMS_EXT} ]] ; then
  i=0
  while [[ -e ${user_params_file}-$i${PARAMS_EXT} || -L ${user_params_file}-$i${PARAMS_EXT} ]] ; do
    let i++
  done
  user_params_file=${user_params_file}-$i${PARAMS_EXT}
fi
echo "{
\"ConvertEvents_NumberOfCrystals_int\": 31200,
\"ConvertEvents_LUTScannerPath_std::string\": \"modular\",
\"ConvertEvents_ScannerName_std::string\": \"modular\",
\"ConvertEvents_OutputConvertedPath_std::string\": \"${OUTPUT_FOLDER}/castor_file\",
\"ConvertEvents_CrystalSizeX_mm_float\": 24,
\"ConvertEvents_CrystalSizeY_mm_float\": 6,
\"ConvertEvents_CrystalSizeZ_mm_float\": 5,
\"ConvertEvents_ScintillatorLength_mm_float\": 500
}" > ${user_params_file}

function executeCommand {
  $@
  rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
  echo "Exit code[" $@ "]: $rc"
}

function run_JPetToCastor {
  executeCommand "./castor-JPetToCastor -t root -f $FILES -u ${user_params_file} -o $OUTPUT_FOLDER -b"
}

function run_CastorReconstruction {
  executeCommand "./castor-recon -df ${OUTPUT_FOLDER}/castor_file.Cdh -dim 160,160,200 -vox 2.5,2.5,2.5 -sens ${SENSITIVITY_MAP_PATH} -th 0 -it 20:1 -dout ${RECONSTRUCTION_FOLDER}"
}

function run_MLP {
# For MLP we need to create new header file with TOF resolution set to 1
# Currently there is no way to override TOF value from command-line in castor
  executeCommand "cp ${OUTPUT_FOLDER}/castor_file.Cdh ${OUTPUT_FOLDER}/castor_file_mlp.Cdh"
  executeCommand "sed -i 's/^TOF resolution (ps): .*$/TOF resolution (ps): 1/' ${OUTPUT_FOLDER}/castor_file_mlp.Cdh"
  executeCommand "sed -i 's/^List TOF measurement range (ps): .*$/List TOF measurement range (ps): 1/' ${OUTPUT_FOLDER}/castor_file_mlp.Cdh"
# Run actual reconstruction
  executeCommand "./castor-recon -df ${OUTPUT_FOLDER}/castor_file_mlp.Cdh -dim 160,160,200 -vox 2.5,2.5,2.5 -opti MLP -th 0 -it 1:1 -dout ${RECONSTRUCTION_FOLDER}_MLP -sens ${SENSITIVITY_MAP_PATH}"
}

function run_Profiles {
  executeCommand "python image_tools.py --header ${RECONSTRUCTION_FOLDER}/${FILENAME}_it20.hdr --out_file ${PROFILES_FOLDER}/${FILENAME} --disable_display --around_bins 3"
}

run_JPetToCastor
run_CastorReconstruction
run_MLP

run_Profiles

exit 0;
