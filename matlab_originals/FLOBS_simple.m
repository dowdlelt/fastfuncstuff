function [basis_functions, time_vec, results] = FLOBS_simple(params)
% FLOBS_simple - Simplified 5-parameter half-cosine HRF basis generation
%
% Parameters (matching your description):
%   m1: Delay before rise starts (seconds)
%   m2: Time from start of rise to peak (seconds)
%   m3: Time from peak to undershoot minimum (seconds)
%   m4: Time from undershoot minimum back to zero (seconds)
%   c:  Magnitude of undershoot relative to peak (0 to 1)
%
% Usage:
%   [basis, t, results] = FLOBS_simple()  % Use defaults
%   [basis, t, results] = FLOBS_simple(params)  % Custom parameters
%
% Input params structure:
%   m1: [min max] - Delay before rise (default: [0 2])
%   m2: [min max] - Rise to peak time (default: [3 8])
%   m3: [min max] - Peak to undershoot time (default: [3 8])
%   m4: [min max] - Undershoot recovery time (default: [3 8])
%   c:  [min max] - Undershoot magnitude (default: [0.15 0.35])
%   num_basis: Number of basis functions (default: 3)
%   num_samples: Number of HRF candidates (default: 1000)
%   TR: Final output resolution in seconds (default: 3)
%   duration: HRF duration in seconds (default: 32)
%   high_res: High-res sampling for generation (default: 0.05)

if nargin < 1 || isempty(params)
    params = struct();
end

% Defaults
if ~isfield(params, 'm1'), params.m1 = [0 2]; end
if ~isfield(params, 'm2'), params.m2 = [3 8]; end
if ~isfield(params, 'm3'), params.m3 = [3 8]; end
if ~isfield(params, 'm4'), params.m4 = [3 8]; end
if ~isfield(params, 'c'), params.c = [0.15 0.35]; end
if ~isfield(params, 'num_basis'), params.num_basis = 3; end
if ~isfield(params, 'num_samples'), params.num_samples = 1000; end
if ~isfield(params, 'TR'), params.TR = 3; end
if ~isfield(params, 'duration'), params.duration = 32; end
if ~isfield(params, 'high_res'), params.high_res = 0.05; end

fprintf('\n=== Simplified FLOBS: 5-Parameter HRF Basis ===\n');
fprintf('Parameter ranges:\n');
fprintf('  m1 (delay):       [%.2f, %.2f] s\n', params.m1(1), params.m1(2));
fprintf('  m2 (rise):        [%.2f, %.2f] s\n', params.m2(1), params.m2(2));
fprintf('  m3 (to under):    [%.2f, %.2f] s\n', params.m3(1), params.m3(2));
fprintf('  m4 (recovery):    [%.2f, %.2f] s\n', params.m4(1), params.m4(2));
fprintf('  c  (undershoot):  [%.2f, %.2f]\n', params.c(1), params.c(2));
fprintf('Resolution: %.3f s (generation) → %.2f s (output)\n', ...
    params.high_res, params.TR);
fprintf('Generating %d candidates...\n', params.num_samples);

% High-res time vector
time_highres = 0:params.high_res:params.duration;
n_highres = length(time_highres);

% Sample parameters using Latin Hypercube
m1_samp = params.m1(1) + (params.m1(2) - params.m1(1)) * lhsdesign(params.num_samples, 1);
m2_samp = params.m2(1) + (params.m2(2) - params.m2(1)) * lhsdesign(params.num_samples, 1);
m3_samp = params.m3(1) + (params.m3(2) - params.m3(1)) * lhsdesign(params.num_samples, 1);
m4_samp = params.m4(1) + (params.m4(2) - params.m4(1)) * lhsdesign(params.num_samples, 1);
c_samp = params.c(1) + (params.c(2) - params.c(1)) * lhsdesign(params.num_samples, 1);

% Generate HRFs at high resolution
HRF_highres = zeros(params.num_samples, n_highres);

for i = 1:params.num_samples
    HRF_highres(i, :) = simple_halfcos_hrf(time_highres, ...
        m1_samp(i), m2_samp(i), m3_samp(i), m4_samp(i), c_samp(i));
end

fprintf('Running PCA...\n');

% PCA on high-res data
[coeff_highres, ~, latent, ~, explained] = pca(HRF_highres);
basis_highres = coeff_highres(:, 1:params.num_basis)';

% Demean (as FSL does)
for i = 1:params.num_basis
    basis_highres(i,:) = basis_highres(i,:) - mean(basis_highres(i,:));
end

% Ensure first basis has correct sign (peak positive)
if abs(min(basis_highres(1,:))) > abs(max(basis_highres(1,:)))
    basis_highres = -basis_highres;
end

% Downsample if needed
if params.TR > params.high_res
    downsample_factor = round(params.TR / params.high_res);
    time_vec = time_highres(1:downsample_factor:end);
    basis_functions = basis_highres(:, 1:downsample_factor:end);
    HRF_matrix = HRF_highres(:, 1:downsample_factor:end);
else
    time_vec = time_highres;
    basis_functions = basis_highres;
    HRF_matrix = HRF_highres;
end

% Store results
results = struct();
results.HRF_matrix = HRF_matrix;
results.HRF_highres = HRF_highres;
results.basis_highres = basis_highres;
results.time_highres = time_highres;
results.latent = latent;
results.explained = explained;
results.parameters = struct('m1', m1_samp, 'm2', m2_samp, ...
    'm3', m3_samp, 'm4', m4_samp, 'c', c_samp);

% Visualization
fprintf('Creating visualizations...\n');
fig = figure('Name', 'Simple FLOBS Results', 'Position', [100 50 1600 800], ...
    'Color', 'w', 'NumberTitle', 'off');

colors = lines(params.num_basis);

% Panel 1: Sample HRFs
subplot(2, 4, 1);
n_show = min(50, params.num_samples);
plot(time_highres, HRF_highres(1:n_show,:)', 'Color', [0.6 0.6 0.6 0.4]);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('%d Sample HRFs', n_show), 'FontWeight', 'bold');
grid on; box on;

% Panel 2: Mean ± SD
subplot(2, 4, 2);
mean_hrf = mean(HRF_highres, 1);
std_hrf = std(HRF_highres, 0, 1);
plot(time_highres, mean_hrf, 'k-', 'LineWidth', 2.5);
hold on;
fill([time_highres, fliplr(time_highres)], ...
    [mean_hrf + std_hrf, fliplr(mean_hrf - std_hrf)], ...
    'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time (s)'); ylabel('Amplitude');
title('Mean HRF ± SD', 'FontWeight', 'bold');
grid on; box on;

% Panel 3: Scree plot
subplot(2, 4, 3);
n_comp = min(12, length(explained));
bar(1:n_comp, explained(1:n_comp), 'FaceColor', [0.3 0.5 0.8]);
hold on;
plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
xlabel('Component'); ylabel('Variance (%)');
title('Scree Plot', 'FontWeight', 'bold');
grid on; box on;

% Panel 4: Cumulative variance
subplot(2, 4, 4);
plot(1:n_comp, cumsum(explained(1:n_comp)), 'b-o', ...
    'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
hold on;
plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
plot(xlim, [95 95], 'k:', 'LineWidth', 1.5);
xlabel('Num Components'); ylabel('Cumulative (%)');
title(sprintf('%.1f%% with %d basis', ...
    sum(explained(1:params.num_basis)), params.num_basis), 'FontWeight', 'bold');
grid on; box on; ylim([0 100]);

% Panels 5-7: Individual basis functions
for i = 1:min(3, params.num_basis)
    subplot(2, 4, 4 + i);
    plot(time_highres, basis_highres(i,:), 'Color', colors(i,:), 'LineWidth', 2);
    if params.TR > params.high_res
        hold on;
        plot(time_vec, basis_functions(i,:), 'o', 'Color', colors(i,:), ...
            'MarkerSize', 4, 'MarkerFaceColor', colors(i,:));
    end
    xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('Basis %d (%.1f%%)', i, explained(i)), 'FontWeight', 'bold');
    grid on; box on;
end

% Panel 8: All basis together
subplot(2, 4, 8);
hold on;
for i = 1:params.num_basis
    h = plot(time_highres, basis_highres(i,:), 'Color', colors(i,:), ...
        'LineWidth', 2.5, 'DisplayName', sprintf('Basis %d', i));
    if params.TR > params.high_res
        plot(time_vec, basis_functions(i,:), 'o', 'Color', colors(i,:), ...
            'MarkerSize', 4, 'MarkerFaceColor', colors(i,:), 'HandleVisibility', 'off');
    end
end
xlabel('Time (s)'); ylabel('Amplitude');
title('All Basis Functions', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on; box on;

% Summary
fprintf('\n=== Complete ===\n');
fprintf('Candidates: %d, Basis functions: %d\n', params.num_samples, params.num_basis);
fprintf('Total variance explained: %.2f%%\n', sum(explained(1:params.num_basis)));
for i = 1:params.num_basis
    fprintf('  Basis %d: %.2f%%\n', i, explained(i));
end
fprintf('Output: %d timepoints at %.2f s resolution\n', ...
    length(time_vec), time_vec(2) - time_vec(1));
fprintf('\n');

end

%% Helper: half-cosine segment
function y = halfcos(t, ybot, ytop, xleft, xright, flipud)
y = zeros(size(t));
if xright > xleft
    mask = (t > xleft) & (t <= xright);
    t_active = t(mask);
    y(mask) = (ytop - ybot) / 2.0 * flipud .* ...
              cos(2 * pi * (t_active - xleft) / ((xright - xleft) * 2)) + ...
              (ytop - ybot) / 2 + ybot;
end
end

%% Simplified 5-parameter HRF
function hrf = simple_halfcos_hrf(t, m1, m2, m3, m4, c)
% Simplified half-cosine HRF with 5 parameters
%
% Structure:
%   Segment 1: [0, m1]        - Flat delay at 0
%   Segment 2: [m1, m1+m2]    - Rise from 0 to 1 (peak)
%   Segment 3: [m1+m2, m1+m2+m3] - Fall from 1 to -c (undershoot)
%   Segment 4: [m1+m2+m3, m1+m2+m3+m4] - Rise from -c back to 0
%
% Peak is always at amplitude 1.0
% Undershoot is at amplitude -c (where c is positive, typically 0.2-0.4)

hrf = zeros(size(t));

% Segment 1: Delay (stays at 0) - no halfcos needed

% Segment 2: Rise to peak (0 to 1)
% Use flip=-1 so cos starts at -1, giving us rise from bottom to top
hrf = hrf + halfcos(t, 0, 1, m1, m1+m2, -1);

% Segment 3: Fall to undershoot (1 to -c)
% Use flip=1 so cos starts at 1, giving us fall from top to bottom
hrf = hrf + halfcos(t, -c, 1, m1+m2, m1+m2+m3, 1);

% Segment 4: Recovery to baseline (-c to 0)
% Use flip=-1 so cos starts at -1, giving us rise from bottom to top
hrf = hrf + halfcos(t, -c, 0, m1+m2+m3, m1+m2+m3+m4, -1);

end