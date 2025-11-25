
   3dDeconvolve -overwrite \
   -input ~/Dropbox/Data/small_test_r0*.nii.gz                      \
   -polort 3 -float                                                       \
   -noFDR                                                                 \
   -jobs 6                                                               \
   -num_stimts 2                                                          \
        -stim_times 1 ~/Dropbox/Projects/MovieTasks/code_movietasks/data_pilot/preproc/sub-pilot01/ses-01/timing_files/small_data_model/ses01_times.movie.txt 'SPMG1(5)' \
        -stim_label 1 movie                                          \
         -stim_times 2 ~/Dropbox/Projects/MovieTasks/code_movietasks/data_pilot/preproc/sub-pilot01/ses-01/timing_files/small_data_model/ses01_times.prompt.txt 'SPMG1(5)' \
        -stim_label 2 prompt                                          \
 -num_glt 1 \
    -jobs 9                                                                                         \
    -gltsym 'SYM: +movie -prompt' \
	-glt_label 1 movieVprompt\
    -tout -fout -x1D ~/Dropbox/Data/X.xmat.1D \
    --bucket none -x1D_stop