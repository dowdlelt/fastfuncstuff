% Dump the voxel->depth sampling and the estimated depth PSF: the two steps the
% reference driver comments out as too slow and ships as cached .mat files.
%
% The reference computes its label-wise means as
%     for i = 1:nlab, out(i) = nanmean(v(MAT_ind==i)); end
% which is a full pass over a 1.5M-voxel volume per label, ~196k times over, and
% ten times again for the PSF profiles. That does not finish. accumarray gets the
% same quantity in one pass; the loop form is spot-checked below on a subset so
% this stays an oracle and not an assumption.

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

nlab = length(unique(MAT_ind(:)));
assert(max(MAT_ind(:)) == nlab, 'labels are not 1..nlab contiguous');

% (a) the downsampling the driver comments out
EV_vox_r = EVV2(:);
EV_ds = label_nanmean(EV_vox_r, MAT_ind(:), nlab);

% spot-check accumarray against the reference loop on the first 200 labels
chk = zeros(200,1);
for i = 1:200, chk(i) = nanmean(EV_vox_r(MAT_ind==i)); end
same_nan = all(isnan(chk) == isnan(EV_ds(1:200)));
d = max(abs(chk(~isnan(chk)) - EV_ds(~isnan(chk))));
if isempty(d), d = 0; end
fprintf('accumarray vs reference loop, 200 labels: max |diff| = %g\n', d);
assert(same_nan && d < 1e-12, 'accumarray disagrees with the reference loop');

% (b) voxel -> depth sampling, condition 1, K = 7 and 9
perVoxResp = [];
for cond = 1:6
    perVoxResp(cond,:,:) = eval(['squeeze(nanmean(nanmean(ER_avg_cond' num2str(cond) ',2),1));']);
end
md = squeeze(perVoxResp(1,:,:));
md = md(sel_sel,:);
y7 = BOLD_voxels2layers_flipdata(md, depth_map_vox_sel, 7);
y9 = BOLD_voxels2layers_flipdata(md, depth_map_vox_sel, 9);

% (c) the PSF estimate, with the random curvature centres pinned and reported
rng(7);
m = 0.5 + 0.06*randn(10,1);
nsig = (1/(4*(N+1)))^2;
IF = [];
for i = 1:length(m)
   IF(:,:,:,i) = 1./sqrt(2*pi*nsig).*exp(-(EVV2-m(i)).^2./(2.*nsig));
end
IF = reshape(IF,[numel(EVV2),length(m)]);
[EV_vox_s,ind0] = unique(EVV2(:));
IFr = zeros(nlab,size(IF,2));
for j = 1:size(IF,2)
    IFr(:,j) = label_nanmean(IF(:,j), MAT_ind(:), nlab);
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

% (d) a small self-contained case pinning the NaN handling of the label mean
rng(11);
small_v = randn(500,1); small_v(1:37:end) = NaN;
small_l = randi(20,500,1);
small_out = label_nanmean(small_v, small_l, 25);

out = struct('m',full(m),'est',full(est(:)),'EV_ds',full(EV_ds), ...
             'ind_vox_sel0',full(ind_vox_sel(:)-1),'depth_sel',full(depth_map_vox_sel(:)), ...
             'data_sel',full(md),'y7',full(y7),'y9',full(y9), ...
             'kernel7',full(kern{1}),'kernel9',full(kern{2}), ...
             'kernel10',full(kern{3}),'kernel11',full(kern{4}), ...
             'nlab',nlab, ...
             'small_v',full(small_v),'small_l0',full(small_l-1),'small_out',full(small_out));
fid=fopen(fullfile(fileparts(mfilename('fullpath')),'laminar_layers_oracle.json'),'w');
fprintf(fid,'%s',jsonencode(out)); fclose(fid);
fprintf('est = [%g %g]; nlab = %d\n', est(1), est(2), nlab);
disp('wrote laminar_layers_oracle.json');

function out = label_nanmean(v, lab, nlab)
% nanmean of v within each integer label 1..nlab, in one pass.
v = v(:); lab = lab(:);
good = ~isnan(v);
s = accumarray(lab(good), v(good), [nlab 1], @sum, 0);
n = accumarray(lab(good), 1,       [nlab 1], @sum, 0);
out = s ./ n;
out(n == 0) = NaN;
end
