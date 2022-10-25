#!/bin/bash

export pvpath="/Applications/ParaView-5.9.1.app/Contents/bin"
startIter=0
skipIter=10
stopIter=20

############################################## Define function #################################################

# Each processor calls pvsingle
# Check consistency with startIter, stopIter outside this function definition
export SHELL=$(type -p bash)
function pvsingle() {
  base="/Users/someguy/work/CEESD/MirgeCom/Drivers/CEESD-Y2_wall/"

  prefix="y2-cavity-multi"

  runName="smoke_test/viz_data"

  # Run Paraview
  echo "${pvpath}/pvbatch paraview-driver.py $base/$runName $prefix $1"
  ${pvpath}/pvbatch paraview-driver.py $base/$runName $prefix $1
  #${pvpath}/pvbatch paraview-driver.py  $base/$runName $prefix 000000
}
export -f pvsingle # So GNU parallel can see it.

#echo "sequence=" $(seq $startIter $skipIter $stopIter)
#parallel -j $numProcs pvsingle ::: $(seq $startIter $skipIter $stopIter)
#pvsingle 000000
for n in $( seq ${startIter} ${skipIter} ${stopIter} )
do
  #echo "pvsingle ${n}"
  pvsingle $n;
done


