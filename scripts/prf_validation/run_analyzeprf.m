function run_analyzeprf(infile, outfile, nworkers)
% Fit analyzePRF to the vertices export_for_matlab.py wrote, timing the fit.
%
% infile holds `stimulus` (row x col x time) and `data` (runs x 1 cell of
% vertices x time, single), `tr`, and `maxpolydeg`. Settings are analyzePRF's
% recommended ones (seedmode 2: supergrid seeds) with its default model (CSS,
% staged exponent), canonical HRF and polynomial degree -- the configuration
% ffs_pyrf -hrf canonical is meant to reproduce.

% analyzePRF bundles 68 utilities that knkutils also ships; whichever is added
% last wins, so add analyzePRF last and check it stuck (wiki: analyzePRF MATLAB
% path shadowing).
apdir = fullfile(getenv('HOME'), 'Dropbox/Resources/code/matlab_toolboxes/analyzePRF');
addpath(genpath(apdir));
assert(strncmp(which('posrect'), apdir, numel(apdir)), 'posrect shadowed: %s', which('posrect'));
in = load(infile);
nruns = numel(in.data);
stim = repmat({double(in.stimulus)}, 1, nruns);
data = cellfun(@double, in.data(:)', 'UniformOutput', false);

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= nworkers
  delete(pool);
  parpool('local', nworkers);
end

opts = struct('seedmode', 2, 'display', 'off', 'maxpolydeg', double(in.maxpolydeg));
t0 = tic;
results = analyzePRF(stim, data, double(in.tr), opts);
elapsed = toc(t0);
fprintf('analyzePRF: %d vertices, %d runs, %.1f s on %d workers\n', ...
        size(data{1}, 1), nruns, elapsed, nworkers);

% Drop the model-fit time series (huge, and recomputable).
keep = {'ang', 'ecc', 'expt', 'rfsize', 'R2', 'gain', 'resnorms', 'numiters'};
out = struct();
for k = 1:numel(keep)
  if isfield(results, keep{k})
    out.(keep{k}) = results.(keep{k});
  end
end
out.params = results.params;
out.hrf = getcanonicalhrf(double(in.tr), double(in.tr))';  % what analyzePRF used
out.elapsed = elapsed;
out.nworkers = nworkers;
save(outfile, '-struct', 'out', '-v7');
end
