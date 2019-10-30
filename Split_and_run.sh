#! /bin/bash


if [ "$#" -ne 4 ]; then
    echo "Split_and_run.sh Mask OutputFolder NSteps NormImages.pkl"
    exit
fi

Basedir=`pwd`
Mask=${1}
OutputFolder=${2}
NSteps=${3}
Data=`echo ${Basedir}/${4}`
n_voxels=`fslstats ${Mask} -V | awk '{print $1}'`
step=`echo '('${n_voxels}'/'${NSteps}')' | bc`

echo ${n_voxels} ${NSteps} ${step}

#Get data into an easier format
mkdir ${Basedir}/${OutputFolder}

#Prep matlab scripts
for subset in $(seq 0 ${NSteps} | awk '{ printf("%04d\n", $1) }'); do
FirstVoxel=`echo ${subset} '*' ${step} | bc`
LastVoxel=`echo ${FirstVoxel} '+' ${step} | bc`

sed "s/INIT/${FirstVoxel}/" ${Basedir}/GPyModel_5D_alt.py > ${Basedir}/${OutputFolder}/Set_${subset}.py
if [ "${LastVoxel}" -gt "${n_voxels}" ]; then 
sed -i "s/END/${n_voxels}/" ${Basedir}/${OutputFolder}/Set_${subset}.py
else
sed -i "s/END/${LastVoxel}/" ${Basedir}/${OutputFolder}/Set_${subset}.py
fi

sed -i "s/ORDER/${subset}/" ${Basedir}/${OutputFolder}/Set_${subset}.py
sed -i "s;PATH;${Basedir}/${OutputFolder};" ${Basedir}/${OutputFolder}/Set_${subset}.py
sed -i "s;NORM_DATA;${Data};" ${Basedir}/${OutputFolder}/Set_${subset}.py
done


