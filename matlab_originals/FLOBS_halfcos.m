function [basis_functions, time_vec, results] = FLOBS_halfcos(params)
% FLOBS_halfcos - Generate HRF basis functions using half-cosine model and PCA
% Based on FSL's halfcosbasis implementation
%
% Usage:
%   [basis, t, results] = FLOBS_halfcos()  % Use defaults
%   [basis, t, results] = FLOBS_halfcos(params)  % Custom parameters
%
% Input params structure fields:
%   m1: [min max] - Delay before rise starts (default: [0 2])
%   m2: [min max] - Time from start to peak (default: [3 8])
%   m3: [min max] - Time from peak to undershoot (default: [3 8])
%   m4: [min max] - Time from undershoot back to zero (default: [3 8])
%   m5: [min max] - Final tail duration (default: [2 5])
%   c1: [min max] - Initial dip magnitude (default: [0 0.2])
%   c2: [min max] - Undershoot magnitude (default: [0.1 0.4])
%   c3: [min max] - Recovery level (default: [0 0.3])
%   num_basis: Number of basis functions (default: 3)
%   num_samples: Number of HRF candidates (default: 1000)
%   TR: Final output resolution (default: 3)
%   duration: HRF duration in seconds (default: 32)
%   high_res: High-res sampling rate for generation (default: 0.05)
%
% Outputs:
%   basis_functions: [num_basis x time_points] matrix
%   time_vec: Time vector in seconds
%   results: Structure with detailed results

% Set default parameters
if nargin < 1 || isempty(params)
    params = struct();
end

% Apply defaults
if ~isfield(params, 'm1'), params.m1 = [0 2]; end
if ~isfield(params, 'm2'), params.m2 = [3 8]; end
if ~isfield(params, 'm3'), params.m3 = [3 8]; end
if ~isfield(params, 'm4'), params.m4 = [3 8]; end
if ~isfield(params, 'm5'), params.m5 = [0 0]; end
if ~isfield(params, 'c1'), params.c1 = [0 0]; end
if ~isfield(params, 'c2'), params.c2 = [0 0.3]; end
if ~isfield(params, 'c3'), params.c3 = [0 0]; end
if ~isfield(params, 'num_basis'), params.num_basis = 3; end
if ~isfield(params, 'num_samples'), params.num_samples = 1000; end
if ~isfield(params, 'TR'), params.TR = 3; end
if ~isfield(params, 'duration'), params.duration = 32; end
if ~isfield(params, 'high_res'), params.high_res = 0.05; end

fprintf('\n=== FLOBS Half-Cosine: HRF Basis Function Generation ===\n');
fprintf('Parameter ranges:\n');
fprintf('  m1 (delay):     [%.2f, %.2f] s\n', params.m1(1), params.m1(2));
fprintf('  m2 (to peak):   [%.2f, %.2f] s\n', params.m2(1), params.m2(2));
fprintf('  m3 (to under):  [%.2f, %.2f] s\n', params.m3(1), params.m3(2));
fprintf('  m4 (recovery):  [%.2f, %.2f] s\n', params.m4(1), params.m4(2));
fprintf('  m5 (tail):      [%.2f, %.2f] s\n', params.m5(1), params.m5(2));
fprintf('  c1 (init dip):  [%.2f, %.2f]\n', params.c1(1), params.c1(2));
fprintf('  c2 (undershoot):[%.2f, %.2f]\n', params.c2(1), params.c2(2));
fprintf('  c3 (recovery):  [%.2f, %.2f]\n', params.c3(1), params.c3(2));
fprintf('High-res sampling: %.3f s, Output TR: %.2f s\n', params.high_res, params.TR);
fprintf('Generating %d HRF candidates...\n', params.num_samples);

% High-resolution time vector for generation
time_highres = 0:params.high_res:params.duration;
n_highres = length(time_highres);

% Generate random parameter sets using Latin Hypercube Sampling
m1_samp = params.m1(1) + (params.m1(2) - params.m1(1)) * lhsdesign(params.num_samples, 1);
m2_samp = params.m2(1) + (params.m2(2) - params.m2(1)) * lhsdesign(params.num_samples, 1);
m3_samp = params.m3(1) + (params.m3(2) - params.m3(1)) * lhsdesign(params.num_samples, 1);
m4_samp = params.m4(1) + (params.m4(2) - params.m4(1)) * lhsdesign(params.num_samples, 1);
m5_samp = params.m5(1) + (params.m5(2) - params.m5(1)) * lhsdesign(params.num_samples, 1);
c1_samp = params.c1(1) + (params.c1(2) - params.c1(1)) * lhsdesign(params.num_samples, 1);
c2_samp = params.c2(1) + (params.c2(2) - params.c2(1)) * lhsdesign(params.num_samples, 1);
c3_samp = params.c3(1) + (params.c3(2) - params.c3(1)) * lhsdesign(params.num_samples, 1);

% Generate HRF matrix at high resolution
HRF_highres = zeros(params.num_samples, n_highres);

for i = 1:params.num_samples
    HRF_highres(i, :) = halfcos_hrf(time_highres, m1_samp(i), m2_samp(i), ...
        m3_samp(i), m4_samp(i), m5_samp(i), c1_samp(i), c2_samp(i), c3_samp(i));
end

fprintf('Running PCA on high-res HRFs...\n');

% Perform PCA on high-resolution data
[coeff_highres, ~, latent, ~, explained] = pca(HRF_highres);

% Extract basis functions at high resolution
basis_highres = coeff_highres(:, 1:params.num_basis)';

% % Demean basis functions (as FSL does)
% for i = 1:params.num_basis
%     basis_highres(i,:) = basis_highres(i,:) - mean(basis_highres(i,:));
% end

% Check sign of first basis function and flip if needed
if abs(min(basis_highres(1,:))) > abs(max(basis_highres(1,:)))
    basis_highres = -basis_highres;
end

% Downsample to desired TR if needed
if params.TR > params.high_res
    % Downsample basis functions
    downsample_factor = round(params.TR / params.high_res);
    time_vec = time_highres(1:downsample_factor:end);
    basis_functions = basis_highres(:, 1:downsample_factor:end);
    
    % Also downsample HRF samples for plotting
    HRF_matrix = HRF_highres(:, 1:downsample_factor:end);
else
    time_vec = time_highres;
    basis_functions = basis_highres;
    HRF_matrix = HRF_highres;
end

fprintf('Basis functions: %d timepoints at %.2f s resolution\n', ...
    size(basis_functions, 2), time_vec(2) - time_vec(1));

% Store results
results = struct();
results.HRF_matrix = HRF_matrix;
results.HRF_highres = HRF_highres;
results.basis_highres = basis_highres;
results.time_highres = time_highres;
results.latent = latent;
results.explained = explained;
results.parameters = struct('m1', m1_samp, 'm2', m2_samp, 'm3', m3_samp, ...
    'm4', m4_samp, 'm5', m5_samp, 'c1', c1_samp, 'c2', c2_samp, 'c3', c3_samp);

fprintf('Creating visualizations...\n');

% Create comprehensive figure
fig = figure('Name', 'FLOBS Half-Cosine Results', 'Position', [100 50 1400 900], ...
    'Color', 'w', 'NumberTitle', 'off');

% Plot 1: Sample of candidate HRFs (show subset)
subplot(3, 3, 1);
n_show = min(100, params.num_samples);
plot(time_highres, HRF_highres(1:n_show, :)', 'Color', [0.7 0.7 0.7 0.3]);
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title(sprintf('%d Sample Candidate HRFs', n_show), 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% Plot 2: Mean HRF with variability
subplot(3, 3, 2);
mean_hrf = mean(HRF_highres, 1);
std_hrf = std(HRF_highres, 0, 1);
plot(time_highres, mean_hrf, 'k-', 'LineWidth', 2.5);
hold on;
fill([time_highres, fliplr(time_highres)], ...
    [mean_hrf + std_hrf, fliplr(mean_hrf - std_hrf)], ...
    'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title('Mean HRF ± SD', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% Plot 3: Scree plot
subplot(3, 3, 3);
n_components = min(15, length(explained));
bar(1:n_components, explained(1:n_components), 'FaceColor', [0.3 0.5 0.8]);
hold on;
plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
xlabel('Principal Component', 'FontSize', 10);
ylabel('Variance Explained (%)', 'FontSize', 10);
title('Scree Plot', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% Plot 4-6: Individual basis functions (high-res)
colors = lines(params.num_basis);
for i = 1:params.num_basis
    subplot(3, 3, 3 + i);
    plot(time_highres, basis_highres(i, :), 'Color', colors(i, :), 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Amplitude', 'FontSize', 10);
    title(sprintf('Basis %d (%.1f%% var)', i, explained(i)), ...
        'FontSize', 11, 'FontWeight', 'bold');
    grid on; box on;
end

% Plot 7: All basis functions together (with downsampling markers)
subplot(3, 3, 7);
hold on;
for i = 1:params.num_basis
    plot(time_highres, basis_highres(i, :), 'Color', colors(i, :), ...
        'LineWidth', 2, 'DisplayName', sprintf('Basis %d', i));
    % Show downsampled points if applicable
    if params.TR > params.high_res
        plot(time_vec, basis_functions(i, :), 'o', 'Color', colors(i, :), ...
            'MarkerSize', 4, 'MarkerFaceColor', colors(i, :), ...
            'HandleVisibility', 'off');
    end
end
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
if params.TR > params.high_res
    title(sprintf('Basis Functions (line=%.2fs, dots=%.2fs)', ...
        params.high_res, params.TR), 'FontSize', 11, 'FontWeight', 'bold');
else
    title('HRF Basis Functions', 'FontSize', 11, 'FontWeight', 'bold');
end
legend('Location', 'best', 'FontSize', 9);
grid on; box on;

% Plot 8: Cumulative variance explained
subplot(3, 3, 8);
n_cumul = min(20, length(explained));
plot(1:n_cumul, cumsum(explained(1:n_cumul)), 'b-o', ...
    'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
hold on;
plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
plot(xlim, [95 95], 'k:', 'LineWidth', 1.5);
xlabel('Number of Components', 'FontSize', 10);
ylabel('Cumulative Variance (%)', 'FontSize', 10);
title(sprintf('Cumulative Variance (%.1f%% with %d)', ...
    sum(explained(1:params.num_basis)), params.num_basis), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;
ylim([0 100]);

% Plot 9: Parameter distributions
subplot(3, 3, 9);
param_data = [m1_samp, m2_samp, m3_samp, m4_samp, m5_samp];
boxplot(param_data, 'Labels', {'m1', 'm2', 'm3', 'm4', 'm5'}, ...
    'Colors', 'k', 'Symbol', 'k.');
ylabel('Time (s)', 'FontSize', 10);
title('Timing Parameter Distribution', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% Print summary
fprintf('\n=== Analysis Complete ===\n');
fprintf('Number of HRF candidates: %d\n', params.num_samples);
fprintf('Number of basis functions: %d\n', params.num_basis);
fprintf('Variance explained: %.2f%%\n', sum(explained(1:params.num_basis)));
fprintf('\nIndividual component variance:\n');
for i = 1:params.num_basis
    fprintf('  Basis %d: %.2f%%\n', i, explained(i));
end
fprintf('\n');

end

%% Half-cosine function (matches FSL implementation)
function y = halfcos(t, ybot, ytop, xleft, xright, flipud)
% Half-cosine wave from xleft to xright, ybot to ytop
% flipud: 1 for normal (cos starts at 1), -1 for flipped (cos starts at -1)

y = zeros(size(t));

if xright > xleft
    mask = (t > xleft) & (t <= xright);
    t_active = t(mask);
    
    % Half-cosine formula from FSL
    y(mask) = (ytop - ybot) / 2.0 * flipud .* ...
              cos(2 * pi * (t_active - xleft) / ((xright - xleft) * 2)) + ...
              (ytop - ybot) / 2 + ybot;
end

end

%% Generate HRF using half-cosine basis (matches FSL)
function hrf = halfcos_hrf(t, m1, m2, m3, m4, m5, c1, c2, c3)
% Generate HRF using 5 half-cosine segments
%
% Segments:
%   1: [0, m1]                    - Initial dip: -c1 to 0, flip=1
%   2: [m1, m1+m2]                - Rise to peak: -c1 to 1, flip=-1
%   3: [m1+m2, m1+m2+m3]          - Fall to undershoot: 1 to -c2, flip=1
%   4: [m1+m2+m3, m1+m2+m3+m4]    - Recovery: -c2 to c3, flip=-1
%   5: [m1+m2+m3+m4, m1+m2+m3+m4+m5] - Final tail: c3 to 0, flip=1

hrf = zeros(size(t));

% Segment 1: Initial dip
hrf = hrf + halfcos(t, -c1, 0, 0, m1, 1);

% Segment 2: Rise to peak
hrf = hrf + halfcos(t, -c1, 1, m1, m1+m2, -1);

% Segment 3: Fall to undershoot
hrf = hrf + halfcos(t, -c2, 1, m1+m2, m1+m2+m3, 1);

% Segment 4: Recovery from undershoot
hrf = hrf + halfcos(t, -c2, c3, m1+m2+m3, m1+m2+m3+m4, -1);

% Segment 5: Final tail to zero
hrf = hrf + halfcos(t, 0, c3, m1+m2+m3+m4, m1+m2+m3+m4+m5, 1);

end