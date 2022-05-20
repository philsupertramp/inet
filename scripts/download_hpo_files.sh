#!/usr/bin/env bash

BASE_DIR="thesisphilipp/model-selection/"
REMOTE_HOST="pzettl@141.64.103.49"

for directory in $(<hpo-directories.txt);
do
 target=$(echo "${directory}" | sed 's/\// /g' | awk '{print $1}');
 target_dir=$(echo "${directory}" | sed 's/\// /g' | awk '{print $2}');

 mkdir -p hpo-files/${target};

 # copy main file
 source_directory="${REMOTE_HOST}:${BASE_DIR}${directory}";
 sshpass -p xvLBJ5qT scp ${source_directory}/oracle.json hpo-files/${target}/main-oracle.json ;

 # copy trial files
 for trial_dir in $(sshpass -p xvLBJ5qT ssh ${REMOTE_HOST} find ${BASE_DIR} -regex \".*/${target_dir}/[^/]+\" -type d);
 do
   trial_id="$(echo $trial_dir | sed 's/\// /g' | awk '{print $5}')"
   sshpass -p xvLBJ5qT scp ${REMOTE_HOST}:${trial_dir}/trial.json hpo-files/${target}/${trial_id}.json
 done

done
