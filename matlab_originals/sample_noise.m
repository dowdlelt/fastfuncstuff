% Demo: Generate and visualize fMRI noise
TR = 1.0; % seconds
duration = 360; % seconds

% Generate noise
noise = generate_fmri_noise(TR, duration);

% Plot time series
figure('Position', [100 100 1200 800]);

subplot(2,1,1);
t = (0:length(noise)-1) * TR;
plot(t, noise, 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Signal (A.U.)');
title('Generated fMRI Noise Time Series');
grid on;

% Plot power spectrum
subplot(2,1,2);
[pxx, f] = pwelch(noise, [], [], [], 1/TR);
plot(f, pxx, 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power');
title('Power Spectrum');
xlim([0 1.5]);
grid on;
hold on;

% Mark the physiological frequencies
xline(0.35, '--r', 'Respiration', 'LineWidth', 1.5);
xline(1.0, '--b', 'Cardiac', 'LineWidth', 1.5);
set(gca, 'YScale', 'log');

% You can customize it:
noise_custom = generate_fmri_noise(TR, duration, ...
    'resp_freq', 0.3, ...
    'resp_strength', 5, ...
    'cardiac_freq', 0.9, ...
    'cardiac_strength', 8);