#!/bin/bash
#PBS -l nodes=8
#PBS -l walltime=02:00:00
#PBS -A CSC188

echo box > SESSION.NAME
echo `pwd`'/' >> SESSION.NAME

cd /lustre/atlas/scratch/kekezhai/csc188/bone-newcurtainTestcase/3dbox-32768-128-lb
rm box.sch
sleep 5


aprun -n 128 -N 16 ./nek5000 > outputscript.txt-n8-71-0.4-32k-lb

