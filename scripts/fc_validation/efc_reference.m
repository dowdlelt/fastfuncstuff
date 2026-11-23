% Reference edge communities from Faskowitz et al. (2020), edge-centric_demo.
% Called by validate.py (stage efc) with `demo` and `out_mat` defined:
%   matlab -batch "demo='/path/edge-centric_demo'; out_mat='ref.mat'; run('efc_reference.m')"
% Five single k-means runs (the demo's call) and one best-of-10, plus the demo's
% node entropy and community-profile similarity from the best-of-10 labels.
addpath(genpath(fullfile(demo, 'fcn')));
load(fullfile(demo, 'example_data.mat'));
ts = double(ts); [T, N] = size(ts);
ets = fcn_edgets(ts);
k = 10; CI = zeros(size(ets, 2), 6); SUMD = zeros(1, 6);
tic;
for s = 1:5
    rng(s); [ci, ~, sumd] = kmeans(ets', k, 'distance', 'sqeuclidean', 'maxiter', 1000);
    CI(:, s) = ci; SUMD(s) = sum(sumd);
end
t_single = toc / 5;
tic;
rng(99);
[ci, ~, sumd] = kmeans(ets', k, 'distance', 'sqeuclidean', 'maxiter', 1000, 'Replicates', 10);
t_rep = toc; CI(:, 6) = ci; SUMD(6) = sum(sumd);
mat = zeros(N); mat(triu(ones(N), 1) > 0) = ci; mat = mat + mat';
S = fcn_profilesim(mat);
[u, v] = find(triu(ones(N), 1));
[~, enorm] = fcn_node_entropy(ci, u, v, N);
save(out_mat, 'CI', 'SUMD', 'S', 'enorm', 'u', 'v', 't_single', 't_rep', '-v7');
