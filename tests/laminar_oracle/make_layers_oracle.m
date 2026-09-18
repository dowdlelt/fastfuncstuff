% Dump the voxel->depth sampling and the estimated depth PSF, the two steps the
% reference driver comments out as too slow and ships as cached .mat files.
REF = '/home/logan/Dropbox/Resources/code/matlab_toolboxes/predictive_tones';
addpath(genpath(fullfile(REF,'Code_laminar_BOLD_model')));
addpath(genpath(fullfile(REF,'Code_layering')));
load(fullfile(REF,'PP_LH_testData.mat'));
load(fullfile(REF,'PP_LH_layer_dist_testData.mat'));
load(fullfile(REF,'ds_vox_depth_PP_LH.mat'));

N = 3;
mat_ind = reshape(1:ceil(nx0/2)*ceil(ny0/2)*ceil(nz0/2),ceil(nx0/2),ceil(ny0/2),ceil(nz0/2));
MAT_ind = [];
for i = 1:ceil(nz0/2)
    MAT_ind = cat(3,MAT_ind,repmat(kron(mat_ind(:,:,i),ones(2)),[1,1,2]));
end
MAT_ind = MAT_ind(1:size(EVV2,1),1:size(EVV2,2),1:size(EVV2,3));

ind_vox = MAT_ind(mask_indices(:,3));
sel     = EV_vox_d(ind_vox);
sel_sel = sel>=0.0005 & sel<=0.9995;
ind_vox_sel = ind_vox(sel_sel);
depth_map_vox_sel = EV_vox_d(ind_vox_sel);

% (a) the downsampling loop the driver comments out
EV_vox_r = EVV2(:);
nlab = length(unique(MAT_ind(:)));
EV_ds = zeros(nlab,1);
for i = 1:nlab, EV_ds(i) = nanmean(EV_vox_r(MAT_ind==i)); end

% (b) voxel -> depth sampling, condition 1, K = 7 and 9
perVoxResp = [];
for cond = 1:6
    perVoxResp(cond,:,:) = eval(['squeeze(nanmean(nanmean(ER_avg_cond' num2str(cond) ',2),1));']);
end
md = squeeze(perVoxResp(1,:,:)); md = md(sel_sel,:);
y7 = BOLD_voxels2layers_flipdata(md, depth_map_vox_sel, 7);
y9 = BOLD_voxels2layers_flipdata(md, depth_map_vox_sel, 9);

% (c) the PSF estimate, with the random curvature centres pinned and reported
rng(7); m = 0.5 + 0.06*randn(10,1);
nsig = (1/(4*(N+1)))^2;
IF = [];
for i = 1:length(m)
   IF(:,:,:,i) = 1./sqrt(2*pi*nsig).*exp(-(EVV2-m(i)).^2./(2.*nsig));
end
IF = reshape(IF,[numel(EVV2),length(m)]);
[EV_vox_s,ind0] = unique(EVV2(:));
IFr = zeros(nlab,size(IF,2));
for j = 1:size(IF,2)
    for i = 1:nlab, IFr(i,j) = nanmean(IF(MAT_ind==i,j)); end
end
[dist, ind] = sort(EV_vox_d(ind_vox_sel));
yy = IFr(ind_vox_sel,:); yy = yy(ind,:);
IFm = interp1(EV_vox_s,IF(ind0,:),dist'); IFm(isnan(IFm)) = 0;
est = fminsearch(@(p)ConvKernel(p,yy,IFm,dist),[1 0.1]);
[err, kernel, yp, sp] = ConvKernel(est,yy,IFm,dist);
kern = {};
for K = [7 9 10 11]
    la = linspace(0,1,2*K+1); la = la(2:2:end);
    ek = interp1(sp,kernel,la,'pchip','extrap'); ek = ek./sum(ek);
    kern{end+1} = ek(:);
end

out = struct('m',m,'est',est(:),'EV_ds',EV_ds, ...
             'ind_vox_sel0',ind_vox_sel(:)-1,'depth_sel',depth_map_vox_sel(:), ...
             'data_sel',md,'y7',y7,'y9',y9, ...
             'kernel7',kern{1},'kernel9',kern{2},'kernel10',kern{3},'kernel11',kern{4}, ...
             'nlab',nlab);
fid=fopen(fullfile(fileparts(mfilename('fullpath')),'laminar_layers_oracle.json'),'w');
fprintf(fid,'%s',jsonencode(out)); fclose(fid);
fprintf('est = [%g %g]; nlab = %d\n', est(1), est(2), nlab);
disp('wrote laminar_layers_oracle.json');
