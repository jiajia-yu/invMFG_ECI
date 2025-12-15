clear;
clc;
% close all;

fontsize = 20;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
flag_plot = false;

%% cases
% filename = 'results/cvg2_mono';
% casename = 'test1';
% vis = 0.1; tend = 1; f_frho = @(x,t,rho) rho;
% casename = 'test2';
% vis = 0.15; tend = 1; f_frho = @(x,t,rho) rho;
% casename = 'test3';
% vis = 0.1; tend = 1.5; f_frho = @(x,t,rho) rho;

filename = 'results/cvg2_nonmono';
% casename = 'test1';
% vis = 0.1; tend = 1; f_frho = @(x,t,rho) -rho;
casename = 'test2';
vis = 1; tend = 1; f_frho = @(x,t,rho) -rho;

%% domain
lend = 0; rend = 1; 
nx = 1000; nt = 200*tend;
opts = [];
opts.step = @(nit) 0.5;
opts.xrange = [lend,rend];
opts.tend = tend;
opts.nx = nx;
opts.nt = nt;
opts.vis_num = 0;
% opts.bd = 'dirichlet';
opts.bd = 'periodic';
% opts.bd = 'neumann';

opts.Nit = 100;
% opts.back_track = true;
opts.tol = 1e-8;
opts.verbose = false;
opts.check_res = true;

opts.ntit = 5;
opts.nttol = 1e-4;
opts.hjb_verbose = false;
opts.fp_verbose = false;

nxp = nx+1; dx = (rend-lend)/nx;
ntp = nt+1; dt = tend/nt;
xgrid = (lend:dx:rend)';
tgrid = (0:dt:tend);
[tmesh,xmesh] = meshgrid(tgrid,xgrid);

%% common settings
% gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
verbose = true;

gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(p) 1/gamma * abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);

f_rho0 = @(x) ones(size(x))/(rend-lend); 
f_q = @(x) -2+cos(2*pi*x) + cos(4*pi*x) + cos(8*pi*x);
% f_q = @(x) 0.1*(exp(sin(2*pi*x.^3))+(x+1).*(x-1).*(x-0.5)-2);
% f_q = @(x) (sin(exp(2*x.^2))+(x+1).*(x-1).*(x-0.5));
% f_q = @(x) ...
%     (x < 0.4 | x > 0.7) .* ( sin(20*pi*x) .* exp(-10*(x - 0.5).^2) ) + ...
%     (x > 0.4 & x < 0.7) .* (-exp(x)) + ...                           
%     (x > 0.7) .* (0.2*sin(100*pi*x)) + ...           
%     (x > 0.3 & x < 0.35) * (-1) + (x > 0.6 & x < 0.65) * (1);
f_g = @(x,rho) zeros(size(x));

% operators
op_lap = @(u) cat(1, u(end-1,  :) - 2*u(1,      :) + u(2,    :), ...
                     u(1:end-2,:) - 2*u(2:end-1,:) + u(3:end,:), ...
                     u(  end-1,:) - 2*u(  end,  :) + u(  2,  :) )/(dx^2); 
op_grd = @(u) cat(1, u(2,     :) - u(end-1,  :), ...
                     u(3:end,:) - u(1:end-2,:), ...
                     u(  2,  :) - u(  end-1,:) )/(2*dx);
op_ham = @(u) func_H(op_grd(u));

norm_sp = @(u) sqrt(sum(u.^2,'all')*dx);
norm_spt = @(u) sqrt(sum(u.^2,'all')*dx*dt);

%% forward problem
rho0 = f_rho0(xgrid); 

q_tru = f_q(xgrid); %q_tru = q_tru - min(q_tru);
q_tru_norm_sp = norm_sp(q_tru);
f_f = @(x,t,rho) q_tru + f_frho(x,t,rho);

tic
[rho_tru,phi_tru,outs_tru] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
t_run = toc;

res_tru = outs_tru.gaps_dual(end);
disp(['forward solver residue = ', num2str(res_tru)]);

%--- measurement
phi0_tru = phi_tru(:,1);
dtphi0_tru = (phi_tru(:,2)-phi_tru(:,1))/dt;

phi0_norm_sp = norm_sp(phi0_tru);
term0_phi = -vis*op_lap(phi0_tru) + op_ham(phi0_tru);

cst_int = dx/tend/(rend-lend);
term0_phi = term0_phi  + sum(phi0_tru)*cst_int;

%% one step comparison
f_qinit = @(x) zeros(size(x));
q_init = f_qinit(xgrid);

f_f = @(x,t,rho) q_init + f_frho(x,t,rho);

tic
[rho_init,phi_init,outs_init] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
t_run = toc;

res_init = outs_init.gaps_dual(end);
disp(['forward solver residue = ', num2str(res_init)]);

phi0_init = phi_init(:,1);
dtphi0_init = (phi_init(:,2)-phi_init(:,1))/dt;

%---
diff_q = q_tru - q_init;
diff_dtphi = dtphi0_tru - dtphi0_init;
diff_intphi = (sum(phi0_tru)-sum(phi0_init))*dx/tend/(rend-lend);
e_phi = diff_intphi + diff_dtphi;
pec = (diff_q).*(diff_q + e_phi);

%% inverse, ECI
flag_cvg = false;

opts.Nit = 100;
% opts.back_track = false;
if isfield(opts,'rho_num')rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')rmfield(opts,'v_num'); end

Nit = 100;
tol = 1e-9;
q_nm1_eci = q_init;
% rho_nm1_eci = rho_init;

q_hist_eci = zeros(nxp,Nit);
nits_eci = zeros(Nit+1,1);
gaps_eci = zeros(Nit,1);
errs_l2_eci = zeros(Nit+1,1);
ress_l2_eci = zeros(Nit,1);

err_nm1_eci = q_tru - q_nm1_eci;
err_nm1_l2_eci = norm_sp(err_nm1_eci)/q_tru_norm_sp;
errs_l2_eci(1) = err_nm1_l2_eci;

disp('ECI iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_eci)])

tic
for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_eci + f_frho(x,t,rho);
    [rho_n_eci,phi_n_eci,outs_n_eci] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_eci;
    opts.v_num = outs_n_eci.v_num;

    phi0_n_eci = phi_n_eci(:,1);
    gdphi_n_eci = op_grd(phi_n_eci);
    term0_phi_n_eci = -vis*op_lap(phi0_n_eci) + func_H(gdphi_n_eci(:,1)) ...
                  + sum(phi0_n_eci)*cst_int;

    % update q    
    res_phi_eci = term0_phi - term0_phi_n_eci;
    q_n_eci = q_nm1_eci + res_phi_eci;

    % record residue, error, number of iteration in ficplay
    nits_eci(nit+1) = nits_eci(nit) + length(outs_n_eci.gaps_dual);
    gaps_eci(nit) = outs_n_eci.gaps_dual(end);
    ress_l2_eci(nit) = norm_sp(phi0_n_eci-phi0_tru)/phi0_norm_sp;

    err_n_eci = q_tru - q_n_eci;
    err_n_l2_eci = norm_sp(err_n_eci)/q_tru_norm_sp;
    errs_l2_eci(nit+1) = err_n_l2_eci;

    % update index
    q_nm1_eci = q_n_eci;
    % rho_nm1_eci = rho_n_eci;
    err_nm1_eci = err_n_eci;

    q_hist_eci(:,nit) = q_n_eci;
    
    if ress_l2_eci(nit) < tol
        flag_cvg = true;
        q_hist_eci = q_hist_eci(:,1:nit);
        nits_eci = nits_eci(1:nit+1);
        gaps_eci = gaps_eci(1:nit);
        ress_l2_eci = ress_l2_eci(1:nit);
        errs_l2_eci = errs_l2_eci(1:nit+1);
        disp(['ECI converges at ',num2str(nit),'-th iteration'])
        break
    end

end
t_run_inv = toc;

if ~flag_cvg
    disp(['ECI finishes at ',num2str(nit),'-th iteration'])
end

save([filename,'_',casename]);

% %% save results
% if ~isfile([filename,'.mat'])
%     save(filename,'filename','xgrid','q_tru','q_init') 
% end
% 
% S = struct();
% S.(['tend_', casename]) = tend;
% S.(['phi0_tru_', casename]) = phi0_tru;
% S.(['phi0_init_', casename]) = phi0_init;
% S.(['dtphi0_tru_', casename]) = dtphi0_tru;
% S.(['dtphi0_init_', casename]) = dtphi0_init;
% S.(['e_phi_', casename]) = e_phi;
% S.(['pec_', casename]) = pec;
% S.(['errs_', casename]) = errs_l2_eci;
% S.(['ress_', casename]) = ress_l2_eci;
% 
% save(filename, '-struct', 'S', '-append');

