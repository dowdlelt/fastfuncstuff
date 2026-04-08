

% the NORDIC Dicom best possible world Run
cd('/home/logan/Dropbox/Resources/code/fastfuncstuff/test_data/nordic_test_data')
addpath(genpath('~/Dropbox/Resources/code/matlab_toolboxes/NORDIC_Raw'));

% Lets clean up a bit and run with better settings. 

% If all are identical 
ARGA.temporal_phase=1;
ARGA.phase_filter_width=10; % From your settings, may not matter
ARGA.noise_volume_last = 3; % last three volumes are noise vols
ARGA.NORDIC =1; % Do NORDIC 
ARGA.save_gfactor_map= 1;
% ARGA.gfactor_patch_overlap=6;
% ARGA.kernel_size=[28 28 1];



fn_magn_in=sprintf('sub-3003_ses-fine_task-expres_run-1_part-mag_bold.nii.gz'); % magnitude data
fn_phase_in = sprintf('sub-3003_ses-fine_task-expres_run-1_part-phase_bold.nii.gz'); % phase data
fn_out=['NORDIC_' fn_magn_in(1:end-7)];
NIFTI_NORDIC(fn_magn_in,fn_phase_in,fn_out,ARGA)

