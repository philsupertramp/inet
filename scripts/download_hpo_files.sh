#!/usr/bin/env bash
# Helper to download HPO history files
# Requires:
#       create ./hpo-directories.txt a file holding the directories (starting from BASE_DIR)
#       to export the HPO results from
#       Example content:
#       clf-custom-cnn/Classification-Custom-CNN-HPO
#       clf-mobilenet/Classification-MobileNet-HPO
# Usage:
#       env REMOTE_HOST=[REMOTE HOST] SSH_PASS=[REMOTE HOST PASSWORD] bash ./scripts/download_hp_files.sh


BASE_DIR="thesisphilipp/model-selection/"

for directory in $(<hpo-directories.txt);
do
 target=$(echo "${directory}" | sed 's/\// /g' | awk '{print $1}');
 target_dir=$(echo "${directory}" | sed 's/\// /g' | awk '{print $2}');

 mkdir -p hpo-files/${target};

 # copy main file
 source_directory="${REMOTE_HOST}:${BASE_DIR}${directory}";
 sshpass -p ${SSH_PASS} scp ${source_directory}/oracle.json hpo-files/${target}/main-oracle.json ;

 # copy trial files
 for trial_dir in $(sshpass -p ${SSH_PASS} ssh ${REMOTE_HOST} find ${BASE_DIR} -regex \".*/${target_dir}/[^/]+\" -type d);
 do
   trial_id="$(echo $trial_dir | sed 's/\// /g' | awk '{print $5}')"
   sshpass -p ${SSH_PASS} scp ${REMOTE_HOST}:${trial_dir}/trial.json hpo-files/${target}/${trial_id}.json
 done

done
