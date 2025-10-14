addpath(genpath('~/Dropbox/Resources/code/matlab_toolboxes/GLMsingle-ltd/matlab'))
addpath(genpath('~/Dropbox/Resources/code/matlab_toolboxes/GLMdenoise-ltd'))
fancy_hrfs = 0

tr =1;
totalSdur = 290;
stimSdur = 5;
n_conds = 2;
blank_trials = 0;
nRuns = 4;


target_mean = 4; % what is the average ISI you want?
alternate_conds = 1;

totalTRdur = round(totalSdur/tr,0);
padding = round(10/tr,0);
stimTrdur = round((totalSdur - 20)/tr,0);

data = cell(1,nRuns);
design=cell(1,nRuns);

stimTRdur = round(stimSdur/tr, 0);
%% Build weird library, mayve
if fancy_hrfs
    params.m3 = [3 10];
    params.m4 = [3 12 ];
    params.c2 = [0 0.35];
    params.m1 = [0 2];
    params.m2 = [3 8];
    params.TR = 1

    [basis_functions, time_vec, results] = FLOBS_halfcos(params)
    [library, ~, ~ ] = create_HRF_library(results)

    % This is at the highresolution sampling rate - 0.05.

    uprate = params.TR/0.05;

    % Create a boxcar that respects that, then we will downsample.
    boxcar = zeros(1,30*uprate);
    boxcar(1,1:5*uprate) = 1;

    highres_hrfs = [];
    ready_hrfs = [];

    % Create convolution.
    for hrf_ix = 1:20
        highres_hrfs(hrf_ix,:) = conv2(boxcar,library(hrf_ix, :) );
        highres_hrfs(hrf_ix,:) = highres_hrfs(hrf_ix,:)/max(highres_hrfs(hrf_ix,:));
        % downsample

        ready_hrfs(hrf_ix,:) = highres_hrfs(hrf_ix,1:uprate:end);
    end

    ready_hrfs = ready_hrfs(:,1:60);

else
    ready_hrfs = getcanonicalhrflibrary(stimSdur,tr);
end


%% Build the onset vector
% %--- Setup Constants ---

block_average = target_mean + stimSdur;
n_target = floor((totalSdur-(2*padding))/block_average);
while mod(n_target,n_conds)
    n_target=n_target -1 ;
end


block_dur_s = ((stimSdur*n_conds)+(n_conds*target_mean))*n_target;
lower_limit = 2; % k >= 2
upper_limit = 8; % k <= 7
if target_mean > upper_limit
    disp('check yer bounds')
    pause
end

max_iter = 100;
tolerance = 0.01; % Mean tolerance
conv=0;
% --- Search Parameters for lambda (Initial Guess) ---
lambda_low = 1.0;
lambda_high = 10.0;
lambda_opt = (lambda_low + lambda_high) / 2;

% --- Iterative Search for Optimal Lambda ---
for iter = 1:max_iter

    % --- Stage 1: Generate Data for the current lambda_opt ---
    % Generate more than n_target to ensure we can select the required count
    % using the truncation criteria.

    % The loop needs to be fixed to ensure the count is met:
    Sonsets = [];
    while numel(Sonsets) < n_target
        % Generate a large batch to speed things up
        new_batch = poissrnd(lambda_opt, 1, n_target * 5);

        % Apply truncation
        truncated = new_batch(new_batch >= lower_limit & new_batch <= upper_limit);

        Sonsets = [Sonsets truncated];
    end

    % Trim to exactly n_target elements
    Sonsets = Sonsets(1:n_target);

    % --- Stage 2: Check Mean and Adjust Lambda (Bisection Method) ---
    current_mean = mean(Sonsets);

    % Check for convergence
    if abs(current_mean - target_mean) < tolerance
        disp(['Converged! Optimal lambda: ', num2str(lambda_opt), ' Mean: ', num2str(current_mean)]);
        conv = 1;
        break;
    end

    % Adjust lambda based on the error
    if current_mean < target_mean
        % If the mean is too low, we need a larger lambda
        lambda_low = lambda_opt;
    else
        % If the mean is too high, we need a smaller lambda
        lambda_high = lambda_opt;
    end

    % Optional: print progress
    disp(['Iteration ', num2str(iter), ': Lambda = ', num2str(lambda_opt), ', Mean = ', num2str(current_mean)]);
    % Update lambda_opt
    lambda_opt = (lambda_low + lambda_high) / 2;


end

if conv == 0
    disp('Max iterations reached without convergence.');
end

onsets = round(Sonsets/tr,0);
%% build task
% Distribute Magnitudes
% Lets make these kind of meaningful.
%5 is good 7t study
% 11 for 3T
ab_ixs = [
    5,1
    5,2
    5,3
    5,4
    5,5
    4,5
    3,5
    2,5
    1,5
    1,3
    3,3
    3,1
    1,1
    0,2
    2,0
    -1,-1
    -3, -3
    -3,4
    4,-3
    0, 0];



for runnum = 1:nRuns
    % lets generate the onsets for different runs
    if blank_trials>0
        trials = randperm(numel(onsets));
        trials(1:blank_trials) = 0;
    end

    shuffled_isis = onsets(randperm(numel(onsets)));
    actualonsets = cumsum([shuffled_isis(1), shuffled_isis(2:end)+stimTRdur])+padding;

    ons_ix = actualonsets;

    temp1 = zeros(totalTRdur,n_conds);
    if alternate_conds % this will just go A/B or A/B/C simple
        for cond_ix = 1:n_conds
            temp1(ons_ix(:,cond_ix:n_conds:end), cond_ix) = 1;
        end
    else
        % TODO not yet implemented - ewould be random onsets (a lot more
        % computation)
    end


    design{1,runnum} =temp1;

    % along the X dimension, 5 steps (so 100 elements) so 1 -5, HRF1, 6 -10
    % HRF2, etc
    % Y dimension, a different ratio of A to B. have to think about this..
    % along the slice dimension, different noise levels.
    temp_data = zeros(100,100,10, totalTRdur);
    fprintf("making slices...")
    for slice_ix = 1:25
        fprintf('.')
        curr_hrf = 1;
        for xdim_ix = 1:100
            ab_ix = 1;

            for ydim_ix = 1:100
                if mod(ydim_ix-1, 5) == 0 & ydim_ix-1 ~=0
                    % change values
                    ab_ix = ab_ix+1;

                end
                % if a_ix ==13
                %     %I want to see
                %     disp(13)
                % end

                ons_vec = zeros(1,ons_ix(end));
                for cond_ix = 1:n_conds
                    ons_vec(ons_ix(cond_ix:n_conds:end)) = ab_ixs(ab_ix,cond_ix);
                end
                signal = conv2(ons_vec,ready_hrfs(curr_hrf,:));
                signal = [signal(1:stimTrdur+padding) zeros(1,padding)];

                temp_data(xdim_ix, ydim_ix, slice_ix, :) = 100+ signal + (rand(1, totalTRdur)*(slice_ix-1));
            end

            if mod(xdim_ix-1, 5) == 0 & xdim_ix-1 ~=0
                xdim_ix;

                curr_hrf = curr_hrf+1;
            end
        end
        noise = generate_fmri_noise(tr, totalTRdur, 'matrix_size', [100 100]);
        temp_data(:,:,slice_ix,:) = temp_data(:,:,slice_ix,:) + permute(noise,[2,3, 4, 1]);
    end
    data{runnum} = single(temp_data);
    fprintf("done\n")
end

%% GLMsingle

opt.wantpercentbold = 1;
opt.hrfdesign =1;
opt.wantglmdenoise=0;
opt.wantfracridge = 0;
opt.wantfileoutputs = [0,1,0,0];
opt.chunknum = 250000
GLMestimatesingletrial(design, data,stimSdur, tr, 'test_movie_mean3', opt)
load('test_movie_mean3/TYPEB_FITHRF.mat');
R2_old = R2;


opt.hrfdesign =2;
GLMestimatesingletrial(design, data,stimSdur, tr, 'test2_5min', opt)
load('test2_5min/TYPEB_FITHRF.mat');

figure; plot(squeeze(R2_old(:,12,1))); hold on
plot(squeeze(R2(:,12,1)));


%% Alternative
opt.numpcstotry = 0;
opt.numboots = 0;
[results,~] = GLMdenoisedata(design,data,stimSdur,tr,'fir', [30],opt,'fir_test_5min_run');

%% inspect figures
% first - we will make a figure that shows if we recovered the input hrfs.


curr_hrf = 1;
model_choice = 3;
model_ix = model_choice*5;

for slice_ix =1
    figure
    curr_hrf = 1;
    model_choice = 16;
    model_ix = model_choice*5;
    hrf_plot=1;
    for xdim_ix = 1:100
        if mod(xdim_ix-1, 5) == 0 & xdim_ix-1 ~=0
            % increment subplot
            curr_hrf = curr_hrf+1;
            hrf_plot = 1;
        end
        subplot(4,5,curr_hrf)
        hold on ;
        for cond_ix = 1:n_conds
            if ab_ixs(model_choice,cond_ix) ==0
                plot(squeeze(results.modelmd(xdim_ix,model_ix,slice_ix,cond_ix,:)));
            else

                plot(squeeze(results.modelmd(xdim_ix,model_ix,slice_ix,cond_ix,:)./ab_ixs(model_choice,cond_ix)));
            end

        end
        if hrf_plot

            plot(ready_hrfs(curr_hrf,1:30),'LineWidth',3)
            hrf_plot = 0;
        end
        ylim([-.5, 1.5]);

    end
end

%% Lets make a version that plots conditions seperately

for slice_ix =8
    figure
    curr_hrf = 1;
    model_choice = 1;
    model_ix = model_choice*5;
    hrf_plot=2;
    for xdim_ix = 50:55
        curr_hrf = round(xdim_ix/50,0)*10;
        % if mod(xdim_ix-1, 5) == 0 & xdim_ix-1 ~=0
        %     % increment subplot
        %     curr_hrf = curr_hrf+1;
        %     hrf_plot = 2;
        % end

        for cond_ix = 1:n_conds
            subplot(1,n_conds,cond_ix)
            hold on ;

            plot(squeeze(results.modelmd(xdim_ix,model_ix,slice_ix,cond_ix,:)));
            ylim([-1.5, 1.5]*max(abs(ab_ixs(model_choice,:))));
            if hrf_plot > 0

                plot(ready_hrfs(curr_hrf,1:30)*ab_ixs(model_choice,cond_ix),'LineWidth',3 ,'Color','k')
                hrf_plot = hrf_plot -1;
            end

        end



    end
end

%% get numbers

corrcoef(squeeze(results.modelmd(xdim_ix,model_ix:model_ix,slice_ix,1,:)./ab_ixs(model_choice,1)) ...
    ,ready_hrfs(curr_hrf,1:31))

