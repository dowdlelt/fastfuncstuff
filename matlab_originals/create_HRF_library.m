function [library, lib_params, lib_info] = create_HRF_library(results, method, n_hrfs)
% CREATE_HRF_LIBRARY - Create a library of representative HRFs
%
% Usage:
%   [library, params, info] = create_HRF_library(results, method, n_hrfs)
%
% Inputs:
%   results - Output structure from FLOBS_simple or FLOBS_halfcos
%   method  - Selection method:
%             'kmeans' - K-means clustering (default, most representative)
%             'pca_reconstruct' - Reconstruct using varying PC components
%             'parameter_grid' - Sample evenly across parameter space
%             'variance_weighted' - Select based on PCA score variance
%             'random' - Random selection from candidates
%   n_hrfs  - Number of HRFs for library (default: 20)
%
% Outputs:
%   library    - [n_hrfs x timepoints] matrix of HRF shapes
%   lib_params - Structure with parameters for each library HRF
%   lib_info   - Additional information about selection method

if nargin < 2 || isempty(method)
    method = 'kmeans';
end

if nargin < 3 || isempty(n_hrfs)
    n_hrfs = 20;
end

fprintf('\n=== Creating HRF Library ===\n');
fprintf('Method: %s\n', method);
fprintf('Number of HRFs: %d\n', n_hrfs);

% Get data
HRF_matrix = results.HRF_highres;  % Use high-res version
time_vec = results.time_highres;
n_samples = size(HRF_matrix, 1);
n_timepoints = size(HRF_matrix, 2);

% Initialize outputs
library = zeros(n_hrfs, n_timepoints);
lib_params = struct();
lib_info = struct('method', method, 'n_hrfs', n_hrfs);

switch lower(method)
    
    case 'kmeans'
        %% K-means clustering - finds most representative shapes
        fprintf('Running k-means clustering...\n');
        
        [idx, centroids, sumd] = kmeans(HRF_matrix, n_hrfs, ...
            'Distance', 'sqeuclidean', ...
            'Replicates', 10, ...
            'MaxIter', 500);
        
        library = centroids;
        
        % Find the actual HRF closest to each centroid
        lib_params.indices = zeros(n_hrfs, 1);
        lib_params.cluster_sizes = zeros(n_hrfs, 1);
        lib_params.m1 = zeros(n_hrfs, 1);
        lib_params.m2 = zeros(n_hrfs, 1);
        lib_params.m3 = zeros(n_hrfs, 1);
        lib_params.m4 = zeros(n_hrfs, 1);
        
        for i = 1:n_hrfs
            cluster_members = find(idx == i);
            lib_params.cluster_sizes(i) = length(cluster_members);
            
            % Find closest actual HRF to centroid
            distances = sum((HRF_matrix(cluster_members, :) - centroids(i, :)).^2, 2);
            [~, min_idx] = min(distances);
            actual_idx = cluster_members(min_idx);
            
            lib_params.indices(i) = actual_idx;
            
            % Store parameters if available
            if isfield(results.parameters, 'm1')
                lib_params.m1(i) = results.parameters.m1(actual_idx);
                lib_params.m2(i) = results.parameters.m2(actual_idx);
                lib_params.m3(i) = results.parameters.m3(actual_idx);
                lib_params.m4(i) = results.parameters.m4(actual_idx);
                if isfield(results.parameters, 'c')
                    lib_params.c(i) = results.parameters.c(actual_idx);
                end
            end
        end
        
        lib_info.cluster_sizes = lib_params.cluster_sizes;
        lib_info.within_cluster_sum = sumd;
        
        fprintf('Cluster sizes: min=%d, max=%d, mean=%.1f\n', ...
            min(lib_params.cluster_sizes), max(lib_params.cluster_sizes), ...
            mean(lib_params.cluster_sizes));
        
    case 'pca_reconstruct'
        %% Reconstruct HRFs using different numbers of PCs
        fprintf('Reconstructing from PCA components...\n');
        
        % Use mean HRF as base
        mean_hrf = mean(HRF_matrix, 1);
        
        % Get PCA components
        if isfield(results, 'pca_coeff')
            coeff = results.pca_coeff;
            score = results.pca_score;
        else
            [coeff, score, ~] = pca(HRF_matrix);
        end
        
        % Strategy: use increasing numbers of PCs
        n_pcs_list = unique(round(linspace(1, min(50, size(coeff, 2)), n_hrfs)));
        n_pcs_list = n_pcs_list(1:min(n_hrfs, length(n_pcs_list)));
        
        % Sample different positions in PC space
        for i = 1:length(n_pcs_list)
            n_pc = n_pcs_list(i);
            
            % Take a sample from score distribution
            score_sample = prctile(score(:, 1:n_pc), 50 + 30*sin(i*2*pi/n_hrfs), 1);
            
            % Reconstruct
            library(i, :) = mean_hrf + score_sample * coeff(:, 1:n_pc)';
        end
        
        % Fill remaining with different percentiles
        for i = (length(n_pcs_list)+1):n_hrfs
            n_pc = min(10, size(coeff, 2));
            pct = 5 + 90 * (i - length(n_pcs_list)) / (n_hrfs - length(n_pcs_list));
            score_sample = prctile(score(:, 1:n_pc), pct, 1);
            library(i, :) = mean_hrf + score_sample * coeff(:, 1:n_pc)';
        end
        
        lib_info.n_pcs_used = n_pcs_list;
        
    case 'parameter_grid'
        %% Sample evenly across parameter space
        fprintf('Sampling parameter space grid...\n');
        
        if ~isfield(results.parameters, 'm1')
            error('Parameter information not available in results structure');
        end
        
        % Determine number of dimensions
        param_names = fieldnames(results.parameters);
        n_dims = length(param_names);
        
        % Create grid - distribute n_hrfs across dimensions
        n_per_dim = round(n_hrfs^(1/n_dims));
        
        % Get parameter ranges
        param_ranges = struct();
        for i = 1:n_dims
            pname = param_names{i};
            param_ranges.(pname) = [min(results.parameters.(pname)), ...
                                    max(results.parameters.(pname))];
        end
        
        % Create grid samples
        if n_dims == 5  % m1, m2, m3, m4, c
            % Create 5D grid
            grid_vals = cell(n_dims, 1);
            for i = 1:n_dims
                pname = param_names{i};
                grid_vals{i} = linspace(param_ranges.(pname)(1), ...
                                       param_ranges.(pname)(2), n_per_dim);
            end
            
            % Use Latin Hypercube instead for better coverage
            lhs_samples = lhsdesign(n_hrfs, n_dims);
            
            lib_params.indices = zeros(n_hrfs, 1);
            for i = 1:n_dims
                pname = param_names{i};
                lib_params.(pname) = param_ranges.(pname)(1) + ...
                    (param_ranges.(pname)(2) - param_ranges.(pname)(1)) * lhs_samples(:, i);
            end
            
            % Generate HRFs for these parameters
            for i = 1:n_hrfs
                if n_dims == 5
                    library(i, :) = simple_halfcos_hrf_vec(time_vec, ...
                        lib_params.m1(i), lib_params.m2(i), lib_params.m3(i), ...
                        lib_params.m4(i), lib_params.c(i));
                else
                    % Find closest existing HRF
                    param_vec = [lib_params.m1(i), lib_params.m2(i), ...
                                lib_params.m3(i), lib_params.m4(i)];
                    distances = sum((results.parameters.m1 - param_vec(1)).^2 + ...
                                   (results.parameters.m2 - param_vec(2)).^2 + ...
                                   (results.parameters.m3 - param_vec(3)).^2 + ...
                                   (results.parameters.m4 - param_vec(4)).^2);
                    [~, min_idx] = min(distances);
                    lib_params.indices(i) = min_idx;
                    library(i, :) = HRF_matrix(min_idx, :);
                end
            end
        end
        
        lib_info.parameter_ranges = param_ranges;
        
    case 'variance_weighted'
        %% Select based on variance in PCA space
        fprintf('Selecting based on PCA score variance...\n');
        
        if isfield(results, 'pca_score')
            score = results.pca_score;
        else
            [~, score, ~] = pca(HRF_matrix);
        end
        
        % Use first few PCs to define "uniqueness"
        n_pc_use = min(10, size(score, 2));
        score_subset = score(:, 1:n_pc_use);
        
        % Select diverse samples using maximin distance
        lib_params.indices = zeros(n_hrfs, 1);
        
        % Start with sample closest to mean
        mean_score = mean(score_subset, 1);
        distances_to_mean = sum((score_subset - mean_score).^2, 2);
        [~, first_idx] = min(distances_to_mean);
        lib_params.indices(1) = first_idx;
        
        % Iteratively add most different samples
        for i = 2:n_hrfs
            selected_scores = score_subset(lib_params.indices(1:i-1), :);
            
            % For each candidate, find minimum distance to selected set
            min_dists = zeros(n_samples, 1);
            for j = 1:n_samples
                if ismember(j, lib_params.indices(1:i-1))
                    min_dists(j) = -Inf;  % Already selected
                else
                    dists_to_selected = sum((selected_scores - score_subset(j, :)).^2, 2);
                    min_dists(j) = min(dists_to_selected);
                end
            end
            
            % Select sample with maximum minimum distance (most different)
            [~, max_idx] = max(min_dists);
            lib_params.indices(i) = max_idx;
        end
        
        library = HRF_matrix(lib_params.indices, :);
        
        % Store parameters
        if isfield(results.parameters, 'm1')
            for pname = fieldnames(results.parameters)'
                lib_params.(pname{1}) = results.parameters.(pname{1})(lib_params.indices);
            end
        end
        
    case 'random'
        %% Random selection
        fprintf('Random selection...\n');
        
        lib_params.indices = randperm(n_samples, n_hrfs);
        library = HRF_matrix(lib_params.indices, :);
        
        % Store parameters
        if isfield(results.parameters, 'm1')
            for pname = fieldnames(results.parameters)'
                lib_params.(pname{1}) = results.parameters.(pname{1})(lib_params.indices);
            end
        end
        
    otherwise
        error('Unknown method: %s', method);
end

%% Visualization
fprintf('Creating visualization...\n');
visualize_library(library, time_vec, lib_info, HRF_matrix);

fprintf('Library creation complete!\n\n');

end

%% Visualization function
function visualize_library(library, time_vec, lib_info, HRF_all)

n_hrfs = size(library, 1);

fig = figure('Name', sprintf('HRF Library (%s)', lib_info.method), ...
    'Position', [50 50 1600 900], 'Color', 'w', 'NumberTitle', 'off');

% Plot 1: All library HRFs
subplot(2, 3, 1);
colors = parula(n_hrfs);
hold on;
for i = 1:n_hrfs
    plot(time_vec, library(i, :), 'Color', colors(i, :), 'LineWidth', 1.5);
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title(sprintf('Library: %d HRFs', n_hrfs), 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

% Plot 2: Library HRFs with mean ± SD overlay
subplot(2, 3, 2);
mean_hrf = mean(library, 1);
std_hrf = std(library, 0, 1);
hold on;
for i = 1:n_hrfs
    plot(time_vec, library(i, :), 'Color', [0.7 0.7 0.7 0.5], 'LineWidth', 1);
end
plot(time_vec, mean_hrf, 'k-', 'LineWidth', 3);
fill([time_vec, fliplr(time_vec)], ...
    [mean_hrf + std_hrf, fliplr(mean_hrf - std_hrf)], ...
    'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Library Mean ± SD', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

% Plot 3: Compare to all samples
subplot(2, 3, 3);
n_show = min(200, size(HRF_all, 1));
plot(time_vec, HRF_all(1:n_show, :)', 'Color', [0.85 0.85 0.85 0.3]);
hold on;
for i = 1:n_hrfs
    plot(time_vec, library(i, :), 'Color', colors(i, :), 'LineWidth', 2);
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
title('Library vs All Samples', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

% Plot 4: Pairwise correlation matrix
subplot(2, 3, 4);
corr_mat = corr(library');
imagesc(corr_mat);
colorbar;
colormap(jet);
caxis([min(corr_mat(:)), 1]);
xlabel('Library HRF', 'FontSize', 11);
ylabel('Library HRF', 'FontSize', 11);
title('Pairwise Correlations', 'FontSize', 12, 'FontWeight', 'bold');
axis square;

% Plot 5: Coverage analysis
subplot(2, 3, 5);
% For each sample HRF, find closest library HRF
if size(HRF_all, 1) > 0
    min_dists = zeros(size(HRF_all, 1), 1);
    for i = 1:size(HRF_all, 1)
        dists = sum((library - HRF_all(i, :)).^2, 2);
        min_dists(i) = sqrt(min(dists));
    end
    histogram(min_dists, 30, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('Distance to Nearest Library HRF', 'FontSize', 11);
    ylabel('Count', 'FontSize', 11);
    title(sprintf('Coverage (mean dist=%.3f)', mean(min_dists)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
end

% Plot 6: Peak times and undershoot distribution
subplot(2, 3, 6);
peak_times = zeros(n_hrfs, 1);
undershoot_times = zeros(n_hrfs, 1);
peak_amps = zeros(n_hrfs, 1);
undershoot_amps = zeros(n_hrfs, 1);

for i = 1:n_hrfs
    [peak_amps(i), peak_idx] = max(library(i, :));
    peak_times(i) = time_vec(peak_idx);
    
    % Find undershoot (minimum after peak)
    [undershoot_amps(i), under_idx] = min(library(i, peak_idx:end));
    undershoot_times(i) = time_vec(peak_idx + under_idx - 1);
end

scatter(peak_times, undershoot_amps, 100, colors, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Time to Peak (s)', 'FontSize', 11);
ylabel('Undershoot Amplitude', 'FontSize', 11);
title('Library HRF Characteristics', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
colormap(gca, parula);
c = colorbar;
c.Label.String = 'Library Index';

end

%% Helper: vectorized HRF generation
function hrf = simple_halfcos_hrf_vec(t, m1, m2, m3, m4, c)
hrf = zeros(size(t));

% Segment 2: Rise to peak
mask2 = (t > m1) & (t <= m1+m2);
if any(mask2)
    t_rel = t(mask2) - m1;
    hrf(mask2) = hrf(mask2) + 0.5 * (-1) * cos(2*pi*t_rel/(m2*2)) + 0.5;
end

% Segment 3: Fall to undershoot
mask3 = (t > m1+m2) & (t <= m1+m2+m3);
if any(mask3)
    t_rel = t(mask3) - (m1+m2);
    hrf(mask3) = hrf(mask3) + (1+c)/2 * cos(2*pi*t_rel/(m3*2)) + (1-c)/2;
end

% Segment 4: Recovery
mask4 = (t > m1+m2+m3) & (t <= m1+m2+m3+m4);
if any(mask4)
    t_rel = t(mask4) - (m1+m2+m3);
    hrf(mask4) = hrf(mask4) + (-c)/2 * (-1) * cos(2*pi*t_rel/(m4*2)) + (-c)/2;
end
end