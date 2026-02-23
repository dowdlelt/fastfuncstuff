# Curreny issues

## 3dREMLfast.py

For voxels with initial transients, or perhaps a lot of drift not fully removed by polynomials (think CSF, cardiac voxels) the t-stats are subtlely differnt. Overall, agreement between 3dREMLfit and ffs is high, but even when using identical matrix and AFNI REMLVar and double precision, they are no mathetmatically identical. 
Level of Concern: Low, these are non significant, the t-stat effect is aroudn .5 at most, and the beta difference is a fraction of the mean signal in that voxel. still weird tho. 
