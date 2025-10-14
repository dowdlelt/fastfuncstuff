function noise_ts = generate_fmri_noise(TR, duration_s, varargin)
% Generate realistic fMRI noise with 1/f spectrum + physiological components
%
% INPUTS:
%   TR          - Repetition time in seconds
%   duration_s  - Duration of scan in seconds
%   
% Optional name-value pairs:
%   'matrix_size'    - Size of output [rows, cols] (default: [1, 1])
%   'fs_high'        - High sampling rate for generation (default: 10 Hz)
%   'resp_freq'      - Respiratory frequency in Hz (default: 0.35)
%   'resp_width'     - Respiratory peak width (default: 0.1)
%   'resp_strength'  - Respiratory peak strength (default: 3)
%   'cardiac_freq'   - Cardiac frequency in Hz (default: 1.0)
%   'cardiac_width'  - Cardiac peak width (default: 0.05)
%   'cardiac_strength' - Cardiac peak strength (default: 5)
%   'pink_exp'       - 1/f exponent (default: 1, can be 0.5-1.5)
%   'normalize'      - Normalize output to unit variance (default: true)
%
% OUTPUT:
%   noise_ts - Time series [n_trs, rows, cols] or [n_trs, 1] if matrix_size=[1,1]

% Parse inputs
p = inputParser;
addParameter(p, 'matrix_size', [1, 1]);
addParameter(p, 'fs_high', 10);
addParameter(p, 'resp_freq', 0.35);
addParameter(p, 'resp_width', 0.1);
addParameter(p, 'resp_strength', 3);
addParameter(p, 'cardiac_freq', 1.0);
addParameter(p, 'cardiac_width', 0.05);
addParameter(p, 'cardiac_strength', 5);
addParameter(p, 'pink_exp', 1.0);
addParameter(p, 'normalize', true);
parse(p, varargin{:});

matrix_size = p.Results.matrix_size;
fs_high = p.Results.fs_high;
resp_freq = p.Results.resp_freq;
resp_width = p.Results.resp_width;
resp_strength = p.Results.resp_strength;
cardiac_freq = p.Results.cardiac_freq;
cardiac_width = p.Results.cardiac_width;
cardiac_strength = p.Results.cardiac_strength;
pink_exp = p.Results.pink_exp;
do_normalize = p.Results.normalize;

% Calculate total voxels
n_voxels = prod(matrix_size);

% Generate at high sampling rate
n_samples = round(duration_s * fs_high);
dt = 1/fs_high;

% Create frequency vector
freqs = (0:n_samples-1) * fs_high / n_samples;
freqs(freqs > fs_high/2) = freqs(freqs > fs_high/2) - fs_high;
freqs = abs(freqs);

% Initialize power spectrum (same for all voxels)
power_spectrum = zeros(n_samples, 1);

% 1/f (pink noise) component
pink_power = 1 ./ (freqs(:) + 0.01).^pink_exp;
power_spectrum = power_spectrum + pink_power;

% Respiratory component (broad Gaussian peak)
resp_component = resp_strength * exp(-((freqs(:) - resp_freq).^2) / (2 * resp_width^2));
power_spectrum = power_spectrum + resp_component;

% Cardiac component (narrow Gaussian peak)
cardiac_component = cardiac_strength * exp(-((freqs(:) - cardiac_freq).^2) / (2 * cardiac_width^2));
power_spectrum = power_spectrum + cardiac_component;

% Convert power to amplitude
amplitude_spectrum = sqrt(power_spectrum);

% Generate random phases for ALL voxels at once - this is where independence comes from
phases = 2*pi*rand(n_samples, n_voxels) - pi;

% Replicate amplitude spectrum for all voxels
amplitude_spectrum_all = repmat(amplitude_spectrum, 1, n_voxels);

% Create complex spectrum for all voxels
complex_spectrum = amplitude_spectrum_all .* exp(1i * phases);

% Ensure conjugate symmetry for real output (for each voxel independently)
complex_spectrum(1, :) = real(complex_spectrum(1, :)); % DC component
if mod(n_samples, 2) == 0
    % Even length
    complex_spectrum(n_samples/2 + 1, :) = real(complex_spectrum(n_samples/2 + 1, :)); % Nyquist
    complex_spectrum(n_samples/2+2:end, :) = conj(complex_spectrum(n_samples/2:-1:2, :));
else
    % Odd length
    complex_spectrum((n_samples+3)/2:end, :) = conj(complex_spectrum((n_samples+1)/2:-1:2, :));
end

% Inverse FFT along first dimension (time/frequency) for ALL voxels at once
noise_high = real(ifft(complex_spectrum, [], 1));

% Downsample to target TR
fs_target = 1/TR;
downsample_factor = round(fs_high / fs_target);
noise_ts = downsample(noise_high, downsample_factor);

% Trim to exact number of TRs
n_trs = floor(duration_s / TR);
noise_ts = noise_ts(1:n_trs, :);

% Normalize if requested (per voxel)
if do_normalize
    noise_ts = (noise_ts - mean(noise_ts, 1)) ./ std(noise_ts, [], 1);
end

% Reshape to [n_trs, rows, cols]
if n_voxels > 1
    noise_ts = reshape(noise_ts, n_trs, matrix_size(1), matrix_size(2));
else
    noise_ts = noise_ts(:); % Keep as column vector for single voxel
end

end