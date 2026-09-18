% Dump reference values from the predictive_tones MATLAB tree so the PyTorch
% port can be checked against them without MATLAB in the loop.
%
% Run once:  matlab -batch "run('make_oracle.m')"
% Writes laminar_oracle.json next to this script.

REF = '/home/logan/Dropbox/Resources/code/matlab_toolboxes/predictive_tones';
addpath(genpath(fullfile(REF,'Code_laminar_BOLD_model')));
addpath(fullfile(REF,'SPM12_download2019','spm12'));

out = struct();
cases = {};

for K = [7 9]
    N = 3;
    M = LBR_param_priors(N, K, zeros(N,N), zeros(N,N), zeros(N,2));
    % The driver's overrides (apply_laminar_BOLD_model.m)
    M.P0.s = 0; M.P0.V0t = 3; M.P0.nr = 3; M.P0.al_v = 0.35; M.P0.w_v = 0.5;
    M.P0.nb = 0; M.P0.tau_v_same = 1; M.P0.tau_d_same = 1;
    M.K = K; M.N = N;

    P  = spm_unvec(spm_vec(M.P0)*0, M.P0);
    P.A    = zeros(N);
    P.C    = [zeros(N,1), ones(N,1)];
    P.B    = cat(3, zeros(N));
    P.Bmu  = zeros(N,1);
    P.Blam = 0;

    % Deterministic, non-trivial state: linear states small, log states small.
    n = N*4 + K*4;
    x = 0.05*sin((1:n)').*cos(0.3*(1:n)');
    u = [0.0, 1.0];

    clear LBR_model_fx   % drop the `persistent fx` between cases
    [f, dfdx] = LBR_model_fx(x, u, P, M);

    load(fullfile(REF,'PP_LH_PSFkernel.mat'));  % provides `kernel`
    M.kernel = kernel;
    g = LBR_model_gx(x, u, P, M);

    % Also a case with a non-zero modulatory input and a moved s_d, so the
    % parity test exercises B, sigma and the draining-vein slope.
    P2 = P;
    P2.B     = cat(3, diag([0.7 0 0]));
    P2.sigma = 0.2;
    P2.s_d   = 0.5;
    P2.nsig  = 0.3;
    u2 = [1.0, 1.0];
    clear LBR_model_fx
    [f2, dfdx2] = LBR_model_fx(x, u2, P2, M);
    g2 = LBR_model_gx(x, u2, P2, M);

    c = struct('N',N,'K',K,'x',x,'u',u,'f',f(:),'g',g(:), ...
               'dfdx_times_f', dfdx*f(:), ...
               'u2',u2,'f2',f2(:),'g2',g2(:), ...
               'dfdx2_times_f2', dfdx2*f2(:), ...
               'kernel', kernel(:));
    cases{end+1} = c; %#ok<SAGROW>
end

out.cases = cases;
fid = fopen(fullfile(fileparts(mfilename('fullpath')),'laminar_oracle.json'),'w');
fprintf(fid,'%s',jsonencode(out));
fclose(fid);
disp('wrote laminar_oracle.json');
