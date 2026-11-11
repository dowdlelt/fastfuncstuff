function run_orig(repo, cbffile, outfile)
addpath(repo);
cbf = load(cbffile);
K = size(cbf,2);
P = LBR_parameters(K);
P.T = size(cbf,1)*P.dt;
[LBR,Y] = LBR_model(P,cbf);
save(outfile,'LBR','-ascii','-double');
% steady state example from the header
P2 = LBR_parameters(K); c2 = ones(P2.T/P2.dt,K)*1.6; P2.s_d = 0.4;
[L2,Y2] = LBR_model(P2,c2);
ss = L2(end,:); save([outfile '.ss'],'ss','-ascii','-double');
end
