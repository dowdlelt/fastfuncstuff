% run_glmsingle_comparison.m
%
% Run GLMsingle (Types B, C, D) on ds005165 localizer data and save
% comparison outputs for benchmarking against FFS.
%
% Two phases:
%   1. Run GLMsingle (skipped if .mat outputs already exist)
%   2. Export NIfTI files from results (skipped if NIfTIs already exist)
%
% Set rerun=true before calling to force re-running GLMsingle.
% Set reexport=true to force re-exporting NIfTIs from existing .mat.
%
% Prerequisites:
%   - MNI-space 4D data (either variant, checked in order):
%       processing/ffs_mni_resampled_task-localizer_run-{1..5}.nii.gz  (ffs_nwarp)
%       processing/afni_mni_resampled_task-localizer_run-{1..5}.nii.gz (3dNwarpApply)
%     Created by ffs_benchmark glmsingle_prep stage (run_ffs or run_ref).
%   - GLMsingle on the MATLAB path  (addpath(genpath('/path/to/GLMsingle/matlab')))
%   - fracridge on the MATLAB path  (addpath(genpath('/path/to/fracridge')))
%
% Usage (manual):
%   export FFS_BENCHMARK_DATA_DIR=/path/to/ds005165-download
%   matlab -batch "addpath(genpath('/path/to/GLMsingle/matlab')); \
%                  addpath(genpath('/path/to/fracridge')); \
%                  run('/path/to/run_glmsingle_comparison.m')"
%
% NIfTI outputs (in glmsingle/):
%   --- Type B (HRF selection) ---
%   glmsingle_hrf_index.nii.gz     - best HRF index per voxel (1-20)
%   glmsingle_r2_B.nii.gz          - R² map, Type B (percentage 0-100)
%   glmsingle_fithrf_r2.nii.gz     - 4D: R² for each HRF (vol per HRF)
%   glmsingle_betas_B.nii.gz       - 4D: single-trial betas, Type B
%   --- Type C (PC denoising) ---
%   glmsingle_noisepool.nii.gz     - noise pool mask (binary)
%   glmsingle_r2_C.nii.gz          - R² map, Type C
%   glmsingle_betas_C.nii.gz       - 4D: single-trial betas, Type C
%   glmsingle_pcnum.txt            - optimal PC count (scalar)
%   glmsingle_xvaltrend.txt        - CV curve (1+numpcstotry values)
%   --- Type D (fracridge) ---
%   glmsingle_fracvalue.nii.gz     - optimal fraction per voxel
%   glmsingle_r2_D.nii.gz          - R² map, Type D
%   glmsingle_betas_D.nii.gz       - 4D: single-trial betas, Type D
%   glmsingle_scaleoffset.nii.gz   - 4D (2 vols): autoscale params
%   --- Shared ---
%   glmsingle_mask.nii.gz          - brain mask used
%
% Or interactively:
%   rerun = true; run('/path/to/run_glmsingle_comparison.m')

% Check that GLMsingle and fracridge are already on the MATLAB path.
% Add them in your startup.m or before calling this script, e.g.:
%   addpath(genpath('/path/to/GLMsingle/matlab'))
%   addpath(genpath('/path/to/fracridge'))
if isempty(which('GLMestimatesingletrial'))
    error(['GLMestimatesingletrial not found on MATLAB path.\n' ...
           'Add GLMsingle to your path before running:\n' ...
           '  addpath(genpath(''/path/to/GLMsingle/matlab''))']);
end
if isempty(which('fracridge'))
    error(['fracridge not found on MATLAB path.\n' ...
           'Add fracridge to your path before running:\n' ...
           '  addpath(genpath(''/path/to/fracridge''))']);
end

% Set working directory to the benchmark data directory.
% When called by ffs_benchmark, FFS_BENCHMARK_DATA_DIR is set automatically
% and the runner also passes cd() before run() — this is a belt-and-suspenders
% fallback for standalone use.
bmark_data_dir = getenv('FFS_BENCHMARK_DATA_DIR');
if ~isempty(bmark_data_dir)
    cd(bmark_data_dir);
    cd('ds005165-download');  % Subdirectory where data is expected # TODO fix this, shouldn't be hardcoded
end
% If neither is set, the caller must have cd'd here already.

%% Configuration
data_dir = 'processing';
func_dir = fullfile('sub-01', 'ses-01', 'func');
gs_dir = 'glmsingle';  % GLMsingle outputs (NIfTIs, .mat, figures)
outdir = fullfile(gs_dir, 'outputs');
matfile = fullfile(gs_dir, 'glmsingle_comparison.mat');
fig_dir = fullfile(gs_dir, 'figures');
nifti_dir = gs_dir;  % NIfTIs go directly in glmsingle/

tr = 1.5;           % Resampled TR (from 1.75s via ffs_slicetime -resample 1.5)
stimdur = 3.0;       % Each trial is 3 seconds
nruns = 5;
conditions = {'faces', 'objects', 'scenes', 'scrambled', 'bodies'};
ncond = length(conditions);

% Control flags (set before calling this script to override)
if ~exist('rerun', 'var'), rerun = false; end
if ~exist('reexport', 'var'), reexport = false; end

%% ========================================================================
%  PHASE 1: Run GLMsingle (skip if .mat exists)
%  ========================================================================
if ~exist(matfile, 'file') || rerun
    fprintf('=== PHASE 1: Running GLMsingle ===\n');

    %% Load data (prefer ffs_mni_resampled_*, fall back to afni_mni_resampled_*)
    fprintf('Loading MNI-space resampled data...\n');
    data = cell(1, nruns);
    for r = 1:nruns
        fname = fullfile(data_dir, sprintf('ffs_mni_resampled_task-localizer_run-%d.nii.gz', r));
        if ~exist(fname, 'file')
            fname = fullfile(data_dir, sprintf('afni_mni_resampled_task-localizer_run-%d.nii.gz', r));
        end
        fprintf('  Loading %s...\n', fname);
        nii = niftiread(fname);
        data{r} = single(nii);  % X x Y x Z x T
    end

    [nx, ny, nz, ~] = size(data{1});
    vol_size = [nx ny nz];
    fprintf('  Volume: %d x %d x %d, %d runs\n', nx, ny, nz, nruns);

    %% Build design matrices from BIDS events files
    fprintf('Building design matrices...\n');
    design = cell(1, nruns);
    for r = 1:nruns
        events_file = fullfile(func_dir, ...
            sprintf('sub-01_ses-01_task-localizer_run-%d_events.tsv', r));
        T = readtable(events_file, 'FileType', 'text', 'Delimiter', '\t');
        nt = size(data{r}, 4);

        dm = zeros(nt, ncond);
        for c = 1:ncond
            cond_name = conditions{c};
            cond_rows = strcmp(T.trial_type, cond_name);
            onsets = T.onset(cond_rows);
            for ii = 1:length(onsets)
                tr_idx = round(onsets(ii) / tr) + 1;
                if tr_idx >= 1 && tr_idx <= nt
                    dm(tr_idx, c) = 1;
                end
            end
        end

        design{r} = dm;
        n_events = sum(dm(:));
        fprintf('  Run %d: %d timepoints, %d events\n', r, nt, n_events);
    end

    %% Create output directories
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    %% Run GLMsingle
    fprintf('\n=== Running GLMsingle (Types B, C, D) ===\n');

    opt = struct();
    opt.wantlibrary = 1;       % Use HRF library (Type B: FITHRF)
    opt.wantglmdenoise = 1;    % Use GLMdenoise (Type C: FITHRF_GLMDENOISE)
    opt.wantfracridge = 1;     % Use fracridge (Type D: FITHRF_GLMDENOISE_RR)
    opt.wantfileoutputs = [0 1 1 1];  % Save B, C, D to disk
    opt.wantmemoryoutputs = [0 1 1 1]; % Return B, C, D in memory
    opt.numpcstotry = 20;      % Max PCs for GLMdenoise
    opt.fracs = fliplr(.05:.05:1);  % Default fracridge fracs
    opt.wantautoscale = 1;     % Autoscale Type D betas
    opt.wantpercentbold = 1;   % Convert to percent signal change before fitting
    opt.hrfdesign = 1;         % Use single-trial design for HRF fitting

    results = GLMestimatesingletrial(design, data, stimdur, tr, ...
        {outdir, fig_dir}, opt);

    %% Extract comparison data
    fprintf('\nExtracting comparison data...\n');

    % Type B results
    res_B = results{2};
    HRFindex = res_B.HRFindex(:);
    HRFindexrun = reshape(res_B.HRFindexrun, [], nruns);
    FitHRFR2 = reshape(res_B.FitHRFR2, [], size(res_B.FitHRFR2, 4));
    R2_B = res_B.R2(:);
    modelmd_B = reshape(res_B.modelmd, [], size(res_B.modelmd, 4));

    % Type C results
    res_C = results{3};
    noisepool = res_C.noisepool(:);
    pcnum = res_C.pcnum;
    xvaltrend = res_C.xvaltrend;
    pcregressors = res_C.pcregressors;
    pcvoxels = res_C.pcvoxels(:);
    glmbadness = reshape(res_C.glmbadness, [], size(res_C.glmbadness, ndims(res_C.glmbadness)));
    R2_C = res_C.R2(:);
    modelmd_C = reshape(res_C.modelmd, [], size(res_C.modelmd, 4));

    % Type D results
    res_D = results{4};
    FRACvalue = res_D.FRACvalue(:);
    rrbadness = reshape(res_D.rrbadness, [], size(res_D.rrbadness, ndims(res_D.rrbadness)));
    scaleoffset = reshape(res_D.scaleoffset, [], 2);
    R2_D = res_D.R2(:);
    modelmd_D = reshape(res_D.modelmd, [], size(res_D.modelmd, 4));

    % Brain mask
    meanvol = mean(data{1}, 4);
    thresh = prctile(meanvol(:), 99) * 0.1;
    mask = meanvol(:) > thresh;

    %% Save .mat
    fprintf('Saving to %s...\n', matfile);
    save(matfile, ...
        'design', 'stimdur', 'tr', 'vol_size', 'conditions', 'mask', ...
        'HRFindex', 'HRFindexrun', 'FitHRFR2', ...
        'R2_B', 'modelmd_B', ...
        'noisepool', 'pcnum', 'xvaltrend', 'pcregressors', 'pcvoxels', 'glmbadness', ...
        'R2_C', 'modelmd_C', ...
        'FRACvalue', 'rrbadness', 'scaleoffset', ...
        'R2_D', 'modelmd_D', ...
        '-v7.3');

    fprintf('  Type B: HRFindex (%d voxels), modelmd_B (%d x %d)\n', ...
        length(HRFindex), size(modelmd_B));
    fprintf('  Type C: pcnum=%d, xvaltrend length=%d\n', pcnum, length(xvaltrend));
    fprintf('  Type D: FRACvalue (%d voxels), modelmd_D (%d x %d)\n', ...
        length(FRACvalue), size(modelmd_D));
else
    fprintf('=== PHASE 1: Skipping GLMsingle (mat file exists) ===\n');
    fprintf('  %s\n', matfile);
    fprintf('  Set rerun=true to force re-run.\n');
end

%% ========================================================================
%  PHASE 2: Export NIfTI files from .mat (skip if already exported)
%  ========================================================================
% Check if NIfTI exports already exist
nifti_check = fullfile(nifti_dir, 'glmsingle_hrf_index.nii.gz');
if ~exist(nifti_check, 'file') || reexport || rerun
    fprintf('\n=== PHASE 2: Exporting NIfTI files ===\n');

    % Load .mat if not already in memory
    if ~exist('vol_size', 'var')
        fprintf('  Loading %s...\n', matfile);
        load(matfile);
    end

    % Get template NIfTI info from input data (prefer ffs variant, fall back to afni)
    template_file = fullfile(data_dir, 'ffs_mni_resampled_task-localizer_run-1.nii.gz');
    if ~exist(template_file, 'file')
        template_file = fullfile(data_dir, 'afni_mni_resampled_task-localizer_run-1.nii.gz');
    end
    template_info = niftiinfo(template_file);

    % Create output directory
    if ~exist(nifti_dir, 'dir'), mkdir(nifti_dir); end

    nx = vol_size(1); ny = vol_size(2); nz = vol_size(3);

    

    % --- Type B ---
    fprintf('  Type B (HRF selection):\n');
    write_vol(HRFindex, fullfile(nifti_dir, 'glmsingle_hrf_index'), template_info, vol_size);
    write_vol(R2_B, fullfile(nifti_dir, 'glmsingle_r2_B'), template_info, vol_size);
    write_vol4d(FitHRFR2, fullfile(nifti_dir, 'glmsingle_fithrf_r2'), template_info, vol_size);
    write_vol4d(modelmd_B, fullfile(nifti_dir, 'glmsingle_betas_B'), template_info, vol_size);

    % --- Type C ---
    fprintf('  Type C (PC denoising):\n');
    write_vol(double(noisepool), fullfile(nifti_dir, 'glmsingle_noisepool'), template_info, vol_size);
    write_vol(R2_C, fullfile(nifti_dir, 'glmsingle_r2_C'), template_info, vol_size);
    write_vol4d(modelmd_C, fullfile(nifti_dir, 'glmsingle_betas_C'), template_info, vol_size);
    % Scalar + curve as text files
    fid = fopen(fullfile(nifti_dir, 'glmsingle_pcnum.txt'), 'w');
    fprintf(fid, '%d\n', pcnum);
    fclose(fid);
    fprintf('  Saved: glmsingle_pcnum.txt  (pcnum=%d)\n', pcnum);
    dlmwrite(fullfile(nifti_dir, 'glmsingle_xvaltrend.txt'), xvaltrend, ' ');
    fprintf('  Saved: glmsingle_xvaltrend.txt  (%d values)\n', length(xvaltrend));

    % --- Type D ---
    fprintf('  Type D (fracridge):\n');
    write_vol(FRACvalue, fullfile(nifti_dir, 'glmsingle_fracvalue'), template_info, vol_size);
    write_vol(R2_D, fullfile(nifti_dir, 'glmsingle_r2_D'), template_info, vol_size);
    write_vol4d(modelmd_D, fullfile(nifti_dir, 'glmsingle_betas_D'), template_info, vol_size);
    write_vol4d(scaleoffset, fullfile(nifti_dir, 'glmsingle_scaleoffset'), template_info, vol_size);

    % --- Shared ---
    write_vol(double(mask), fullfile(nifti_dir, 'glmsingle_mask'), template_info, vol_size);

    fprintf('\n=== NIfTI export complete ===\n');
    fprintf('  Output directory: %s\n', nifti_dir);
else
    fprintf('\n=== PHASE 2: Skipping NIfTI export (files exist) ===\n');
    fprintf('  %s\n', nifti_dir);
    fprintf('  Set reexport=true to force re-export.\n');
end

fprintf('\n=== Done ===\n');


% Helper: write a 3D volume
    function write_vol(data_flat, fname, info, vol_sz)
        vol = reshape(data_flat, vol_sz);
        info3d = info;
        info3d.ImageSize = double(vol_sz);
        info3d.PixelDimensions = info.PixelDimensions(1:3);
        info3d.Datatype = 'single';
        info3d.BitsPerPixel = 32;
        info3d.Filename = fname;
        if isfield(info3d, 'raw')
            info3d = rmfield(info3d, 'raw');
        end
        niftiwrite(single(vol), fname, info3d, 'Compressed', true);
        fprintf('  Saved: %s\n', fname);
    end

    % Helper: write a 4D volume
    function write_vol4d(data_2d, fname, info, vol_sz)
        % data_2d is (n_voxels, n_vols) — reshape to (nx, ny, nz, n_vols)
        n_vols = size(data_2d, 2);
        vol4d = reshape(data_2d, [vol_sz, n_vols]);
        info4d = info;
        info4d.ImageSize = [double(vol_sz), n_vols];
        info4d.PixelDimensions = [info.PixelDimensions(1:3), info.PixelDimensions(4)];
        info4d.Datatype = 'single';
        info4d.BitsPerPixel = 32;
        info4d.Filename = fname;
        if isfield(info4d, 'raw')
            info4d = rmfield(info4d, 'raw');
        end
        niftiwrite(single(vol4d), fname, info4d, 'Compressed', true);
        fprintf('  Saved: %s  (%d volumes)\n', fname, n_vols);
    end
