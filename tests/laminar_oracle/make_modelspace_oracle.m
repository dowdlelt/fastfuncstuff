% Dump the FULL eight-model space at two vascular resolutions, plus the
% Bayesian parameter average across them. These are the two parity gaps left
% after make_inversion_oracle.m, which covers only two models at one K.
%
% Note the kernel: the driver loads the shipped K=7 PSF and uses it at *every*
% K. Reproduced here, so this compares against what the authors actually ran.
%
% Run:  matlab -batch "run('make_modelspace_oracle.m')"

REF = '/home/logan/Dropbox/Resources/code/matlab_toolboxes/predictive_tones';
addpath(genpath(fullfile(REF,'Code_laminar_BOLD_model')));
addpath(genpath(fullfile(REF,'Code_layering')));
addpath(fullfile(REF,'SPM12_download2019','spm12'));

load(fullfile(REF,'PP_LH_testData.mat'));
load(fullfile(REF,'PP_LH_layer_dist_testData.mat'));
load(fullfile(REF,'ds_vox_depth_PP_LH.mat'));
load(fullfile(REF,'PP_LH_PSFkernel.mat'));         % -> kernel, K=7, used at all K

N = 3; TR = 1.6; dt = 0.05; gap_size = 20;
cmp1 = 1; cmp2 = 3; K_list = [7 9];

mat_ind = reshape(1:ceil(nx0/2)*ceil(ny0/2)*ceil(nz0/2), ...
                  ceil(nx0/2),ceil(ny0/2),ceil(nz0/2));
MAT_ind = []; interp_factor = 2;
for i = 1:ceil(nz0/2)
    MAT_ind = cat(3,MAT_ind,repmat(kron(mat_ind(:,:,i),ones(interp_factor)),[1,1,interp_factor]));
end
MAT_ind = MAT_ind(1:size(EVV2,1),1:size(EVV2,2),1:size(EVV2,3));
ind_vox = MAT_ind(mask_indices(:,3));
sel = EV_vox_d(ind_vox); sel_sel = sel>=0.0005 & sel<=0.9995;
depth_map_vox_sel = EV_vox_d(ind_vox(sel_sel));

perVoxResp = [];
for cond = 1:6
    perVoxResp(cond,:,:) = eval(['squeeze(nanmean(nanmean(ER_avg_cond' num2str(cond) ',2),1));']);
end

% The eight-model space: null, singles, pairs, all three. Same order as
% layer_model_targets(3) on the Python side.
targets = {[0 0 0],[1 0 0],[0 1 0],[0 0 1],[1 1 0],[1 0 1],[0 1 1],[1 1 1]};
names   = {'null','superficial','middle','deep', ...
           'superficial+middle','superficial+deep','middle+deep', ...
           'superficial+middle+deep'};

all_out = struct('N',N,'TR',TR,'K_list',K_list,'kernel',full(kernel(:)), ...
                 'names',{names},'targets',{targets});
per_k = {};

for ki = 1:numel(K_list)
    K = K_list(ki);
    for condition = [cmp1 cmp2]
        md = squeeze(perVoxResp(condition,:,:)); md = md(sel_sel,:);
        yy = BOLD_voxels2layers_flipdata(md, depth_map_vox_sel, K);
        yy(1,:) = 0;
        eval(['y' num2str(condition) ' = yy;']);
    end
    rng(20260918);
    mid = []; for ii = 1:K, mid(:,ii) = 0.1*randn(gap_size,1); end
    Y.y = eval(['[y' num2str(cmp1) ';mid;y' num2str(cmp2) '];']);

    onset{1} = [{[47.9]}, {[1.6 46.4]}];
    duration{1} = [{[0.1]}, {[1.6 1.6]}];
    cnam = {'c1m','c2d'};
    mask = zeros(size(Y.y)); mask([2:8,30:36],:) = 1;
    ns = size(Y.y,1); nr = size(Y.y,2);
    rnam = strsplit(num2str(1:K));
    DCM = create_SPM_file_for_DCM2(Y,ns,TR,onset,duration,cnam,Inf,rnam,round(TR/dt));
    DCM.Y.X0 = DCM.Y.X0(:,2:end);

    M = LBR_param_priors(N,K,zeros(N,N),zeros(N,N),zeros(N,2));
    M.P0.s=0; M.P0.V0t=3; M.P0.nr=3; M.P0.al_v=0.35; M.P0.w_v=0.5;
    M.P0.nb=0; M.P0.tau_v_same=1; M.P0.tau_d_same=1;
    M.N=N; M.K=K;
    M.TR=TR; M.delays=ones(1,nr)*(TR/2); M.m=nr; M.n=length(M.x(:));
    M.l=nr; M.dt=DCM.U.dt; M.ns=ns; M.asl=0; M.IS='spm_int_IT';
    M.f=@LBR_model_fx; M.g=@LBR_model_gx;
    M.Mask = reshape(find(mask),7*2,K);
    M.kernel = kernel;

    pE = spm_unvec(spm_vec(M.P0)*0,M.P0);
    spC = spm_unvec(spm_vec(M.P0)*0,M.P0);
    pE.A=zeros(N); pE.C=[zeros(N,1),ones(N,1)]; pE.Bmu=zeros(3,1); pE.Blam=0;
    pE.B=cat(3,zeros(N)); pE.mu=-0.8; pE.lam=1.8;
    pE.tau_d_de=0; pE.tau_d_in=0; pE.s_d=0; pE.nb=0;
    spC.C=[zeros(N,1),ones(N,1)]*exp(0); spC.nsig=exp(-4);
    spC.mu=0; spC.sigma=exp(-2); spC.lam=0;
    spC.Bmu=zeros(3,1); spC.Blam=0;
    spC.tau_d_de=0; spC.tau_d_in=0; spC.al_d=exp(-5); spC.s_d=exp(-1);

    res = {};
    for mi = 1:numel(targets)
        spCm = spC; spCm.B = cat(3, diag(targets{mi})*exp(0.5));
        Mm = M; Mm.pE = pE; Mm.pC = diag(spm_vec(spCm));
        Mm.noprint = 1; Mm.nograph = 1;
        D = DCM; D.M = Mm;
        tic; inv_D = LBR_model_inversion_dyn_mask(D); el = toc;
        res{end+1} = struct('name',names{mi},'F',full(inv_D.F), ...
            'Ep_vec',full(spm_vec(inv_D.Ep)),'Cp',full(inv_D.Cp), ...
            'Eh',full(inv_D.Eh(:)),'Ep_B',full(diag(inv_D.Ep.B(:,:,1))), ...
            'pC_vec',full(diag(Mm.pC)),'seconds',el); %#ok<SAGROW>
        DCM_store{mi,ki} = inv_D; %#ok<SAGROW>
        fprintf('K=%d  %-24s F = %12.6f  (%.1f s)\n', K, names{mi}, inv_D.F, el);
    end
    per_k{ki} = struct('K',K,'y',full(Y.y),'u',full(DCM.U.u),'ns',ns, ...
                       'mask_rows',full(any(mask,2)),'models',{res}); %#ok<SAGROW>
end
all_out.per_k = per_k;

% ---- Bayesian parameter average across K, per model
bpa = {};
for mi = 1:numel(targets)
    P = {}; for ki = 1:numel(K_list), P{ki} = DCM_store{mi,ki}; end
    B = spm_dcm_bpa(P);
    bpa{end+1} = struct('name',names{mi},'Ep_vec',full(spm_vec(B.Ep)), ...
        'Cp',full(B.Cp),'F',full(B.F), ...
        'Ep_B',full(diag(B.Ep.B(:,:,1)))); %#ok<SAGROW>
    fprintf('BPA %-24s F = %12.6f\n', names{mi}, B.F);
end
all_out.bpa = bpa;

fid = fopen(fullfile(fileparts(mfilename('fullpath')),'laminar_modelspace_oracle.json'),'w');
fprintf(fid,'%s',jsonencode(all_out)); fclose(fid);
disp('wrote laminar_modelspace_oracle.json');
