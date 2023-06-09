#!/bin/bash

search_dir=./data/audio/intel
for entry in "$search_dir"/*
do
  #echo $entry
  #out_put=${entry//$find/$replace}
  #out_put= sed 's/$find/$replace/g' $entry
  out_put="${entry/intel/intel2}"
  sox $entry $out_put
  #echo $entry
  echo $out_put
done
