function FLOBS_GUI
    % FLOBS - MATLAB implementation of FSL's FLOBS tool
    % Generates HRF basis functions using parameter variation and PCA
    
    % Create main figure
    fig = figure('Name', 'Make FLOBS - HRF Basis Function Generator', ...
        'Position', [100 100 1000 700], 'MenuBar', 'none', ...
        'NumberTitle', 'off', 'Resize', 'on');
    
    % Default parameters
    params = struct();
    params.m1_min = 0; params.m1_max = 2;
    params.m2_min = 3; params.m2_max = 8;
    params.m3_min = 3; params.m3_max = 8;
    params.m4_min = 3; params.m4_max = 8;
    params.c_min = 0; params.c_max = 0.3;
    params.num_basis = 3;
    params.num_samples = 500; % Number of HRF candidates to generate
    params.TR = 3; % Repetition time in seconds
    params.duration = 32; % HRF duration in seconds
    
    % Create UI components
    createUI();
    updatePreview();
    
    function createUI()
        % Preview axes
        ax_preview = axes('Parent', fig, 'Position', [0.05 0.45 0.55 0.5]);
        title(ax_preview, 'HRF Preview');
        xlabel(ax_preview, 'Time (s)');
        ylabel(ax_preview, 'Amplitude');
        grid(ax_preview, 'on');
        
        % Parameter controls
        y_start = 0.32;
        y_step = 0.035;
        
        createParamControl('m1', y_start, 'Time to onset/undershoot');
        createParamControl('m2', y_start - y_step, 'Time to peak');
        createParamControl('m3', y_start - 2*y_step, 'Time of undershoot');
        createParamControl('m4', y_start - 3*y_step, 'Width of undershoot');
        createParamControl('c', y_start - 4*y_step, 'Undershoot ratio');
        
        % Number of basis functions
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.05 0.10 0.20 0.025], ...
            'String', 'Number of basis functions:', ...
            'HorizontalAlignment', 'left', 'FontSize', 10);
        
        uicontrol('Style', 'edit', 'Units', 'normalized', ...
            'Position', [0.26 0.10 0.08 0.03], ...
            'String', num2str(params.num_basis), ...
            'Callback', @(src,~) updateNumBasis(str2double(src.String)));
        
        % Number of samples
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.05 0.06 0.20 0.025], ...
            'String', 'Number of HRF samples:', ...
            'HorizontalAlignment', 'left', 'FontSize', 10);
        
        uicontrol('Style', 'edit', 'Units', 'normalized', ...
            'Position', [0.26 0.06 0.08 0.03], ...
            'String', num2str(params.num_samples), ...
            'Callback', @(src,~) updateNumSamples(str2double(src.String)));
        
        % Buttons
        uicontrol('Style', 'pushbutton', 'String', 'Preview', ...
            'Units', 'normalized', 'Position', [0.40 0.06 0.10 0.04], ...
            'Callback', @(~,~) updatePreview(), 'FontSize', 11);
        
        uicontrol('Style', 'pushbutton', 'String', 'Go', ...
            'Units', 'normalized', 'Position', [0.51 0.06 0.10 0.04], ...
            'Callback', @(~,~) runFLOBS(), 'FontSize', 11, ...
            'BackgroundColor', [0.9 0.9 1]);
        
        uicontrol('Style', 'pushbutton', 'String', 'Help', ...
            'Units', 'normalized', 'Position', [0.62 0.06 0.10 0.04], ...
            'Callback', @(~,~) showHelp(), 'FontSize', 11);
        
        % Info panel
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.65 0.05 0.32 0.35], ...
            'String', sprintf(['FLOBS - HRF Basis Function Generator\n\n' ...
            'This tool generates HRF basis functions by:\n' ...
            '1. Creating many HRF candidates\n' ...
            '2. Varying parameters within ranges\n' ...
            '3. Running PCA to find basis functions\n\n' ...
            'Adjust parameter ranges and click Go.']), ...
            'HorizontalAlignment', 'left', 'FontSize', 9, ...
            'BackgroundColor', [0.95 0.95 0.95]);
    end

    function createParamControl(param_name, y_pos, tooltip)
        % Label
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.05 y_pos 0.05 0.025], ...
            'String', param_name, 'HorizontalAlignment', 'left', ...
            'FontSize', 10, 'FontWeight', 'bold');
        
        % Min label and input
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.11 y_pos 0.04 0.025], ...
            'String', 'Min', 'HorizontalAlignment', 'right', 'FontSize', 9);
        
        uicontrol('Style', 'edit', 'Units', 'normalized', ...
            'Position', [0.16 y_pos 0.06 0.03], ...
            'String', num2str(params.([param_name '_min'])), ...
            'Callback', @(src,~) updateParam([param_name '_min'], str2double(src.String)), ...
            'TooltipString', tooltip);
        
        % Max label and input
        uicontrol('Style', 'text', 'Units', 'normalized', ...
            'Position', [0.23 y_pos 0.04 0.025], ...
            'String', 'Max', 'HorizontalAlignment', 'right', 'FontSize', 9);
        
        uicontrol('Style', 'edit', 'Units', 'normalized', ...
            'Position', [0.28 y_pos 0.06 0.03], ...
            'String', num2str(params.([param_name '_max'])), ...
            'Callback', @(src,~) updateParam([param_name '_max'], str2double(src.String)), ...
            'TooltipString', tooltip);
    end

    function updateParam(field, value)
        params.(field) = value;
        updatePreview();
    end

    function updateNumBasis(value)
        params.num_basis = max(1, round(value));
    end

    function updateNumSamples(value)
        params.num_samples = max(10, round(value));
    end

    function updatePreview()
        % Generate preview HRF using middle values
        m1 = mean([params.m1_min, params.m1_max]);
        m2 = mean([params.m2_min, params.m2_max]);
        m3 = mean([params.m3_min, params.m3_max]);
        m4 = mean([params.m4_min, params.m4_max]);
        c = mean([params.c_min, params.c_max]);
        
        t = 0:0.1:params.duration;
        hrf = generateHRF(t, m1, m2, m3, m4, c);
        
        axes(ax_preview);
        cla;
        plot(t, hrf, 'k-', 'LineWidth', 2);
        hold on;
        
        % Add reference lines
        plot([m1 m1], ylim, 'r--', 'LineWidth', 1);
        plot([m2 m2], ylim, 'b--', 'LineWidth', 1);
        plot([m3 m3], ylim, 'g--', 'LineWidth', 1);
        plot([m3 m3+m4], [min(hrf) min(hrf)], 'm-', 'LineWidth', 2);
        
        % Add labels
        text(m1, max(hrf)*0.9, 'm1', 'Color', 'r', 'FontSize', 10);
        text(m2, max(hrf), 'm2', 'Color', 'b', 'FontSize', 10);
        text(m3, min(hrf), 'm3', 'Color', 'g', 'FontSize', 10);
        text(m3+m4/2, min(hrf)*1.1, 'm4', 'Color', 'm', 'FontSize', 10);
        text(params.duration*0.9, min(hrf)*0.7, sprintf('c=%.2f', c), ...
            'FontSize', 10);
        
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('HRF Preview (using mid-range parameters)');
        grid on;
        hold off;
    end

    function hrf = generateHRF(t, m1, m2, m3, m4, c)
        % Generate double-gamma HRF
        % Positive gamma function (response)
        d1 = m2 - m1; % delay of response
        d2 = m3 - m1; % delay of undershoot
        
        % Scale parameters for gamma functions
        a1 = d1; % shape parameter for response
        b1 = 1;  % scale parameter
        a2 = d2; % shape parameter for undershoot
        b2 = 1;  % scale parameter
        
        % Ensure positive time values for gamma function
        t_shifted = t - m1;
        t_shifted(t_shifted < 0) = 0;
        
        % Generate response and undershoot
        response = (t_shifted.^(a1-1) .* exp(-t_shifted/b1)) / (gamma(a1) * b1^a1);
        undershoot = (t_shifted.^(a2-1) .* exp(-t_shifted/b2)) / (gamma(a2) * b2^a2);
        
        % Apply width parameter to undershoot
        width_scale = exp(-(t - m3).^2 / (2*m4^2));
        undershoot = undershoot .* width_scale;
        
        % Combine with undershoot ratio
        hrf = response - c * undershoot;
        
        % Normalize
        if max(abs(hrf)) > 0
            hrf = hrf / max(hrf);
        end
        
        % Zero out negative times
        hrf(t < m1) = 0;
    end

    function runFLOBS()
        % Main FLOBS computation
        fprintf('Running FLOBS analysis...\n');
        fprintf('Generating %d HRF candidates...\n', params.num_samples);
        
        % Time vector
        t = 0:params.TR/10:params.duration;
        n_timepoints = length(t);
        
        % Generate random parameter sets using Latin Hypercube Sampling
        m1_samples = params.m1_min + (params.m1_max - params.m1_min) * lhsdesign(params.num_samples, 1);
        m2_samples = params.m2_min + (params.m2_max - params.m2_min) * lhsdesign(params.num_samples, 1);
        m3_samples = params.m3_min + (params.m3_max - params.m3_min) * lhsdesign(params.num_samples, 1);
        m4_samples = params.m4_min + (params.m4_max - params.m4_min) * lhsdesign(params.num_samples, 1);
        c_samples = params.c_min + (params.c_max - params.c_min) * lhsdesign(params.num_samples, 1);
        
        % Generate HRF matrix
        HRF_matrix = zeros(params.num_samples, n_timepoints);
        
        for i = 1:params.num_samples
            HRF_matrix(i, :) = generateHRF(t, m1_samples(i), m2_samples(i), ...
                m3_samples(i), m4_samples(i), c_samples(i));
        end
        
        fprintf('Running PCA...\n');
        
        % Perform PCA
        [coeff, score, latent, ~, explained] = pca(HRF_matrix);
        
        % Extract basis functions
        basis_functions = coeff(:, 1:params.num_basis)';
        
        fprintf('PCA complete. Creating visualizations...\n');
        
        % Create results figure
        results_fig = figure('Name', 'FLOBS Results', ...
            'Position', [150 50 1200 800], 'NumberTitle', 'off');
        
        % Plot 1: All candidate HRFs
        subplot(2, 3, 1);
        plot(t, HRF_matrix', 'Color', [0.7 0.7 0.7 0.3]);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('All %d Candidate HRFs', params.num_samples));
        grid on;
        
        % Plot 2: Mean HRF
        subplot(2, 3, 2);
        mean_hrf = mean(HRF_matrix, 1);
        std_hrf = std(HRF_matrix, 0, 1);
        plot(t, mean_hrf, 'k-', 'LineWidth', 2);
        hold on;
        fill([t, fliplr(t)], [mean_hrf + std_hrf, fliplr(mean_hrf - std_hrf)], ...
            'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('Mean HRF ± SD');
        grid on;
        hold off;
        
        % Plot 3: Explained variance
        subplot(2, 3, 3);
        bar(1:min(10, length(explained)), explained(1:min(10, length(explained))));
        xlabel('Principal Component');
        ylabel('Variance Explained (%)');
        title('Scree Plot');
        grid on;
        
        % Plot 4: Basis functions
        subplot(2, 3, 4);
        colors = lines(params.num_basis);
        hold on;
        for i = 1:params.num_basis
            plot(t, basis_functions(i, :), 'Color', colors(i, :), ...
                'LineWidth', 2, 'DisplayName', sprintf('Basis %d', i));
        end
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('%d HRF Basis Functions', params.num_basis));
        legend('Location', 'best');
        grid on;
        hold off;
        
        % Plot 5: Cumulative variance
        subplot(2, 3, 5);
        plot(1:min(20, length(explained)), cumsum(explained(1:min(20, length(explained)))), ...
            'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
        hold on;
        plot([params.num_basis params.num_basis], ylim, 'r--', 'LineWidth', 2);
        xlabel('Number of Components');
        ylabel('Cumulative Variance Explained (%)');
        title(sprintf('Cumulative Variance (%.1f%% with %d components)', ...
            sum(explained(1:params.num_basis)), params.num_basis));
        grid on;
        hold off;
        
        % Plot 6: Parameter distributions
        subplot(2, 3, 6);
        param_data = [m1_samples, m2_samples, m3_samples, m4_samples, c_samples];
        boxplot(param_data, 'Labels', {'m1', 'm2', 'm3', 'm4', 'c'});
        ylabel('Parameter Value');
        title('Parameter Distribution');
        grid on;
        
        % Print summary
        fprintf('\n=== FLOBS Summary ===\n');
        fprintf('Number of HRF candidates: %d\n', params.num_samples);
        fprintf('Number of basis functions: %d\n', params.num_basis);
        fprintf('Variance explained by %d components: %.2f%%\n', ...
            params.num_basis, sum(explained(1:params.num_basis)));
        fprintf('Individual component variance:\n');
        for i = 1:params.num_basis
            fprintf('  Component %d: %.2f%%\n', i, explained(i));
        end
        
        % Save results to base workspace
        assignin('base', 'FLOBS_basis_functions', basis_functions);
        assignin('base', 'FLOBS_time', t);
        assignin('base', 'FLOBS_explained_variance', explained);
        assignin('base', 'FLOBS_all_HRFs', HRF_matrix);
        assignin('base', 'FLOBS_PCA_coeff', coeff);
        
        fprintf('\nResults saved to workspace:\n');
        fprintf('  FLOBS_basis_functions - The basis set\n');
        fprintf('  FLOBS_time - Time vector\n');
        fprintf('  FLOBS_explained_variance - Variance explained by each PC\n');
        fprintf('  FLOBS_all_HRFs - All candidate HRFs\n');
        fprintf('  FLOBS_PCA_coeff - PCA coefficients\n');
        fprintf('\nAnalysis complete!\n');
    end

    function showHelp()
        helpdlg({...
            'FLOBS - HRF Basis Function Generator', ...
            '', ...
            'This tool creates basis functions for HRF modeling in fMRI.', ...
            '', ...
            'Parameters:', ...
            '  m1: Time to onset/start of response', ...
            '  m2: Time to peak of positive response', ...
            '  m3: Time to undershoot minimum', ...
            '  m4: Width/duration of undershoot', ...
            '  c: Ratio of undershoot to main response (0-1)', ...
            '', ...
            'Workflow:', ...
            '1. Set parameter ranges (Min/Max values)', ...
            '2. Set number of basis functions to extract', ...
            '3. Click Preview to see example HRF', ...
            '4. Click Go to run full analysis', ...
            '', ...
            'The tool will:', ...
            '- Generate many HRF candidates', ...
            '- Perform PCA on the candidate set', ...
            '- Extract the specified number of basis functions', ...
            '- Display comprehensive visualizations', ...
            '- Save results to MATLAB workspace', ...
            }, 'FLOBS Help');
    end
end