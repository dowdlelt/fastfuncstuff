% Dump the internals of ONE Variational Laplace iteration at the prior mean, so
% the Python port can be compared term by term instead of only end to end.
REF = '/home/logan/Dropbox/Resources/code/matlab_toolboxes/predictive_tones';
addpath(genpath(fullfile(REF,'Code_laminar_BOLD_model')));
addpath(fullfile(REF,'SPM12_download2019','spm12'));

O = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')),'laminar_inversion_oracle.json')));
N = O.N; K = O.K; ns = O.ns;
y = reshape(O.y, ns, K);
u = reshape(O.u, [], 2);
kernel = O.kernel;

M = LBR_param_priors(N,K,zeros(N,N),zeros(N,N),zeros(N,2));
M.P0.s=0; M.P0.V0t=3; M.P0.nr=3; M.P0.al_v=0.35; M.P0.w_v=0.5;
M.P0.nb=0; M.P0.tau_v_same=1; M.P0.tau_d_same=1; M.N=N; M.K=K;

pE  = spm_unvec(spm_vec(M.P0)*0,M.P0);
spC = spm_unvec(spm_vec(M.P0)*0,M.P0);
pE.A=zeros(N); pE.C=[zeros(N,1),ones(N,1)]; pE.Bmu=zeros(3,1); pE.Blam=0;
pE.B=cat(3,zeros(N)); pE.mu=-0.8; pE.lam=1.8;
pE.tau_d_de=0; pE.tau_d_in=0; pE.s_d=0; pE.nb=0;
spC.C=[zeros(N,1),ones(N,1)]*exp(0); spC.nsig=exp(-4);
spC.mu=0; spC.sigma=exp(-2); spC.lam=0; spC.Bmu=zeros(3,1); spC.Blam=0;
spC.tau_d_de=0; spC.tau_d_in=0; spC.al_d=exp(-5); spC.s_d=exp(-1);
spC.B = cat(3, diag([1 0 0])*exp(0.5));

M.TR=1.6; M.delays=ones(1,K)*(1.6/2); M.m=K; M.n=length(M.x(:));
M.l=K; M.dt=O.dt; M.ns=ns; M.asl=0; M.IS='spm_int_IT';
M.f=@LBR_model_fx; M.g=@LBR_model_gx; M.kernel=kernel;
mask = zeros(ns,K); mask([2:8,30:36],:)=1;
M.Mask = reshape(find(mask),14,K);
M.pE=pE; M.pC=diag(spm_vec(spC));
U.u = u; U.dt = O.dt;

% ---- replicate the first iteration of spm_nlsi_GN_laminar_mask -------------
ny_masked = length(M.Mask(:)); ns_masked = size(M.Mask,1);
nr = ny_masked/ns_masked;
Q  = spm_Ce(ns_masked*ones(1,nr));
nh = length(Q); nq = ny_masked/length(Q{1});
hE = sparse(nh,1) - log(var(spm_vec(y))) + 4;
ihC = speye(nh,nh)*exp(4);
pC = M.pC; V = spm_svd(pC,0); np = size(V,2);
pCr = V'*pC*V; ipC = inv(pCr);
p = zeros(np,1); Ep = pE; h = hE;

IS = spm_funcheck(inline('spm_int_IT(P,M,U)','P','M','U'));
[dfdp,f] = spm_diff(IS,Ep,M,U,1,{V});
dfdp_msk = [];
for ii = 1:np, dfdp_msk{1,ii} = dfdp{1,ii}(M.Mask); end
dfdp_msk = reshape(spm_vec(dfdp_msk),ny_masked,np);
e = spm_vec(y(M.Mask)) - spm_vec(f(M.Mask));
J = -dfdp_msk;

iS = sparse(0);
for i = 1:nh, iS = iS + Q{i}*(exp(-32) + exp(h(i))); end
S = spm_inv(iS); iS = kron(speye(nq),iS);
Cp = spm_inv(real(J'*iS*J) + ipC);
d = h - hE;
Ch = spm_inv(real(-(-ihC)));   % dFdhh before the h-loop contributions

L1 = spm_logdet(iS)*nq/2 - real(e'*iS*e)/2 - ny_masked*log(8*atan(1))/2;
L2 = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;

out = struct('np',np,'nh',nh,'nq',nq,'ny_masked',ny_masked,'ns_masked',ns_masked, ...
             'hE',full(hE),'Q1_diag',full(diag(Q{1}))','Q1_nnz',nnz(Q{1}), ...
             'Q1_offdiag_nnz', nnz(Q{1} - diag(diag(Q{1}))), ...
             'pC_diag_reduced',full(diag(pCr)),'V_cols',full(V), ...
             'f0',full(f),'e',full(e),'dfdp_msk',full(dfdp_msk), ...
             'L1',full(L1),'L2',full(L2), ...
             'logdet_iS',full(spm_logdet(iS)),'logdet_ipCCp',full(spm_logdet(ipC*Cp)), ...
             'var_y',var(spm_vec(y)));
fid=fopen(fullfile(fileparts(mfilename('fullpath')),'laminar_vlstep_oracle.json'),'w');
fprintf(fid,'%s',jsonencode(out)); fclose(fid);
fprintf('np=%d nh=%d nq=%g ny=%d L1=%.6f L2=%.6f\n',np,nh,nq,ny_masked,L1,L2);
disp('wrote laminar_vlstep_oracle.json');
