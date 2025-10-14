function [basis_functions, time_vec, results] = FLOBS(params)
% FLOBS - Generate HRF basis functions using parameter variation and PCA
%
% Usage:
%   [basis, t, results] = FLOBS()  % Use defaults
%   [basis, t, results] = FLOBS(params)  % Custom parameters
%
% Input params structure fields:
%   m1: [min max] - Time to onset/undershoot (default: [0 2])
%   m2: [min max] - Time to peak (default: [3 8])
%   m3: [min max] - Time to undershoot (default: [3 8])
%   m4: [min max] - Width of undershoot (default: [3 8])
%   c:  [min max] - Undershoot ratio (default: [0 0.3])
%   num_basis: Number of basis functions (default: 3)
%   num_samples: Number of HRF candidates (default: 500)
%   TR: Repetition time in seconds (default: 3)
%   duration: HRF duration in seconds (default: 32)
%
% Outputs:
%   basis_functions: [num_basis x time_points] matrix
%   time_vec: Time vector in seconds
%   results: Structure with detailed results
%
% Example:
%   params.m1 = [0 2];
%   params.m2 = [4 7];
%   params.m3 = [10 16];
%   params.m4 = [2 5];
%   params.c = [0 0.35];
%   params.num_basis = 3;
%   params.num_samples = 1000;
%   [basis, t, results] = FLOBS(params);

% Set default parameters
if nargin < 1 || isempty(params)
    params = struct();
end

% Apply defaults for missing fields
if ~isfield(params, 'm1'), params.m1 = [0 2]; end
if ~isfield(params, 'm2'), params.m2 = [3 8]; end
if ~isfield(params, 'm3'), params.m3 = [3 8]; end
if ~isfield(params, 'm4'), params.m4 = [3 8]; end
if ~isfield(params, 'c'), params.c = [0 0.3]; end
if ~isfield(params, 'num_basis'), params.num_basis = 3; end
if ~isfield(params, 'num_samples'), params.num_samples = 500; end
if ~isfield(params, 'TR'), params.TR = 3; end
if ~isfield(params, 'duration'), params.duration = 32; end

fprintf('\n=== FLOBS: HRF Basis Function Generation ===\n');
fprintf('Parameter ranges:\n');
fprintf('  m1 (onset):     [%.2f, %.2f]\n', params.m1(1), params.m1(2));
fprintf('  m2 (peak):      [%.2f, %.2f]\n', params.m2(1), params.m2(2));
fprintf('  m3 (undershoot):[%.2f, %.2f]\n', params.m3(1), params.m3(2));
fprintf('  m4 (width):     [%.2f, %.2f]\n', params.m4(1), params.m4(2));
fprintf('  c (ratio):      [%.2f, %.2f]\n', params.c(1), params.c(2));
fprintf('Generating %d HRF candidates...\n', params.num_samples);

% Time vector
time_vec = 0:params.TR/10:params.duration;
n_timepoints = length(time_vec);

% Generate random parameter sets using Latin Hypercube Sampling
m1_samples = params.m1(1) + (params.m1(2) - params.m1(1)) * lhsdesign(params.num_samples, 1);
m2_samples = params.m2(1) + (params.m2(2) - params.m2(1)) * lhsdesign(params.num_samples, 1);
m3_samples = params.m3(1) + (params.m3(2) - params.m3(1)) * lhsdesign(params.num_samples, 1);
m4_samples = params.m4(1) + (params.m4(2) - params.m4(1)) * lhsdesign(params.num_samples, 1);
c_samples = params.c(1) + (params.c(2) - params.c(1)) * lhsdesign(params.num_samples, 1);

% Generate HRF matrix
HRF_matrix = zeros(params.num_samples, n_timepoints);

for i = 1:params.num_samples
    HRF_matrix(i, :) = generateHRF(time_vec, m1_samples(i), m2_samples(i), ...
        m3_samples(i), m4_samples(i), c_samples(i));
end

fprintf('Running PCA...\n');

% Perform PCA
[coeff, score, latent, ~, explained] = pca(HRF_matrix);

% Extract basis functions
basis_functions = coeff(:, 1:params.num_basis)';

% Store results
results = struct();
results.HRF_matrix = HRF_matrix;
results.pca_coeff = coeff;
results.pca_score = score;
results.latent = latent;
results.explained = explained;
results.parameters = struct('m1', m1_samples, 'm2', m2_samples, ...
    'm3', m3_samples, 'm4', m4_samples, 'c', c_samples);

fprintf('PCA complete. Creating visualizations...\n');

% Create comprehensive figure
fig = figure('Name', 'FLOBS Results', 'Position', [100 50 1400 900], ...
    'Color', 'w', 'NumberTitle', 'off');

% Plot 1: All candidate HRFs
subplot(3, 3, 1);
plot(time_vec, HRF_matrix', 'Color', [0.7 0.7 0.7 0.2]);
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title(sprintf('All %d Candidate HRFs', params.num_samples), 'FontSize', 11, 'FontWeight', 'bold');
grid on;
box on;

% Plot 2: Mean HRF with variability
subplot(3, 3, 2);
mean_hrf = mean(HRF_matrix, 1);
std_hrf = std(HRF_matrix, 0, 1);
plot(time_vec, mean_hrf, 'k-', 'LineWidth', 2.5);
hold on;
fill([time_vec, fliplr(time_vec)], ...
    [mean_hrf + std_hrf, fliplr(mean_hrf - std_hrf)], ...
    'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title('Mean HRF ± SD', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
box on;

% Plot 3: Scree plot
subplot(3, 3, 3);
n_components = min(15, length(explained));
bar(1:n_components, explained(1:n_components), 'FaceColor', [0.3 0.5 0.8]);
hold on;
plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
xlabel('Principal Component', 'FontSize', 10);
ylabel('Variance Explained (%)', 'FontSize', 10);
title('Scree Plot', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
box on;

% Plot 4: Basis functions (individual)
colors = lines(params.num_basis);
for i = 1:params.num_basis
    subplot(3, 3, 3 + i);
    plot(time_vec, basis_functions(i, :), 'Color', colors(i, :), 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Amplitude', 'FontSize', 10);
    title(sprintf('Basis Function %d (%.1f%% var)', i, explained(i)), ...
        'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    box on;
end

% Plot 5: All basis functions together
subplot(3, 3, 7);
hold on;
for i = 1:params.num_basis
    plot(time_vec, basis_functions(i, :), 'Color', colors(i, :), ...
        'LineWidth', 2.5, 'DisplayName', sprintf('Basis %d', i));
end
xlabel('Time (s)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title(sprintf('%d HRF Basis Functions', params.num_basis), ...
    'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
box on;

% Plot 6: Cumulative variance explained
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
grid on;
box on;
ylim([0 100]);

% Plot 7: Parameter distributions
subplot(3, 3, 9);
param_data = [m1_samples, m2_samples, m3_samples, m4_samples, c_samples];
boxplot(param_data, 'Labels', {'m1', 'm2', 'm3', 'm4', 'c'}, ...
    'Colors', 'k', 'Symbol', 'k.');
ylabel('Parameter Value', 'FontSize', 10);
title('Parameter Distribution', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
box on;

% Print summary
fprintf('\n=== Analysis Complete ===\n');
fprintf('Number of HRF candidates: %d\n', params.num_samples);
fprintf('Number of basis functions: %d\n', params.num_basis);
fprintf('Variance explained by %d components: %.2f%%\n', ...
    params.num_basis, sum(explained(1:params.num_basis)));
fprintf('\nIndividual component variance:\n');
for i = 1:params.num_basis
    fprintf('  Basis %d: %.2f%%\n', i, explained(i));
end
fprintf('\n');

end

%% Helper function to generate single HRF
function hrf = generateHRF(t, m1, m2, m3, m4, c)
% Generate double-gamma HRF based on FSL FLOBS model
% Parameters:
%   t: time vector
%   m1: time to onset
%   m2: time to peak
%   m3: time to undershoot
%   m4: width of undershoot
%   c: undershoot ratio

% Shift time relative to onset
t_shifted = t - m1;
t_shifted(t_shifted < 0) = 0;

% Calculate delays
d1 = m2 + m1;  % delay of response
d2 = m3 + m1 + m1;  % delay of undershoot

% Shape parameters for gamma functions
a1 = max(d1, 0.1);
a2 = max(d2, 0.1);
b1 = 1;
b2 = 1;

% Generate positive response (gamma function)
response = (t_shifted.^(a1-1) .* exp(-t_shifted/b1)) / (gamma(a1) * b1^a1);

% Generate undershoot (gamma function with width modulation)
undershoot_gamma = (t_shifted.^(a2-1) .* exp(-t_shifted/b2)) / (gamma(a2) * b2^a2);

% Apply width parameter using Gaussian envelope centered at m3
if m4 > 0
    width_envelope = exp(-(t - m3).^2 / (2*m4^2));
    undershoot = undershoot_gamma .* width_envelope;
else
    undershoot = undershoot_gamma;
end

% Combine response and undershoot
hrf = response - c * undershoot;

% Normalize to unit peak
if max(abs(hrf)) > 0
    hrf = hrf / max(hrf);
end

% Zero out before onset
hrf(t < m1) = 0;

% Handle any NaN or Inf values
hrf(~isfinite(hrf)) = 0;

end