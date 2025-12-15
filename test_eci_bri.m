clear;
clc;
% close all;
        
fontsize = 20;
linewidth = 2;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
verbose = true;

%% domain
lend = 0; rend = 1; tend = 1;
nx = 1000; nt = 500;
opts = [];
opts.xrange = [lend,rend];
opts.tend = tend;
opts.nx = nx;
opts.nt = nt;
opts.vis_num = 1;
opts.bd = 'periodic';

opts.step = @(nit) 0.5;
opts.back_track = false;
opts.tol = 1e-6;
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

%% true
% gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
vis = 0.3;
gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(p) 1/gamma * abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);

f_rho0 = @(x) gau(0.5,0.5/sqrt(20),x); 
f_q = @(x) 0.1*(sin(2*pi*x-sin(4*pi*x)) + exp(cos(2*pi*x)));
filename = 'compare_ecibri'; 
f_frho = @(x,t,rho) rho.^2;
f_g = @(x,rho)  -f_rho0(x);

% nonlocal interaction
% f_K = @(x,y) exp(- (min(abs(x-y),2-abs(x-y))).^2 /0.05);
% % f_K = @(x,y) exp(- (x-2*y).^2 /0.05);
% K = f_K( xgrid,xgrid' );
% f_frho = @(x,t,rho) K * rho *dx; filename = '1d_nonlocal';

% operators
op_lap = @(u) cat(1, u(end-1,  :) - 2*u(1,      :) + u(2,    :), ...
                     u(1:end-2,:) - 2*u(2:end-1,:) + u(3:end,:), ...
                     u(  end-1,:) - 2*u(  end,  :) + u(  2,  :) )/(dx^2); 
op_grd = @(u) cat(1, u(2,    :) - u(end-1,  :), ...
                     u(3:end,:) - u(1:end-2,:), ...
                     u(  2,  :) - u(  end-1,:) )/(2*dx);
op_ham = @(u) func_H(op_grd(u));

norm_sp = @(u) sqrt(sum(u.^2,'all')*dx);
norm_spt = @(u) sqrt(sum(u.^2,'all')*dx*dt);

%% forward problem
q_tru = f_q(xgrid); %q_tru = q_tru - min(q_tru);
q_tru_norm_sp = norm_sp(q_tru);
f_f = @(x,t,rho) q_tru + f_frho(x,t,rho);
rho0 = f_rho0(xgrid);

tic
[rho_tru,phi_tru,outs_tru] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
t_run = toc;
res_tru = outs_tru.gaps_dual(end);

% save('test_inverse','rho_tru','phi_tru')
disp(['forward solver residue = ', num2str(res_tru)]);

figure(4);clf;
set(gcf,'unit','centimeters','Position',[3 30 24 7]);
subplot(131);plot(xgrid,q_tru,'LineWidth',linewidth);title('true obstacle')
subplot(132);mesh(xmesh,tmesh,phi_tru);xlabel('x');ylabel('t');title('true phi')
subplot(133);mesh(xmesh,tmesh,rho_tru);xlabel('x');ylabel('t');title('true rho')


%% measurement, potentially noisy
phi0 = phi_tru(:,1); 
phi0_norm_sp = norm_sp(phi0);
term0_phi = -vis*op_lap(phi0) + op_ham(phi0);
dtphi0 = term0_phi - q_tru - f_frho(xgrid,0,rho0);

cst_int = dx/tend/(rend-lend);
term0_phi = term0_phi  + sum(phi0)*cst_int;

%% inverse general setting
% HJB parameter
opts_hjb = opts;
if isfield(opts,'hjb_verbose') opts_hjb.verbose = opts.hjb_verbose; end
% FP parameter
opts_fp = opts;
if isfield(opts,'fp_verbose') opts_fp.verbose = opts.fp_verbose; end

tol = 1e-9;
f_qinit = @(x) 0.15*ones(size(x));
rho_init = repmat(rho0,1,ntp);
q_init = f_qinit(xgrid);

%% inverse equilibrium correction iteration (ECI)
flag_cvg = false;

opts.Nit = 100;
if isfield(opts,'rho_num') opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num') opts = rmfield(opts,'v_num'); end

Nit = 100;
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
    nits_eci(nit+1) = nits_eci(nit) + length(outs_n_eci.gaps_dual)+1;
    gaps_eci(nit) = outs_n_eci.gaps_dual(end);
    ress_l2_eci(nit) = norm_sp(phi0_n_eci-phi0)/phi0_norm_sp;

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

if ~flag_cvg
    disp(['ECI finishes at ',num2str(nit),'-th iteration'])
end

%% inverse best response iteration (BRI, nit = 1)
flag_cvg = false;

opts.Nit = 1;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 100;
q_nm1_bri1 = q_init;
% rho_nm1_bri1 = rho_init;

q_hist_bri1 = zeros(nxp,Nit);
nits_bri1 = zeros(Nit+1,1);
gaps_bri1 = zeros(Nit,1);
errs_l2_bri1 = zeros(Nit+1,1);
ress_l2_bri1 = zeros(Nit,1);

err_nm1_bri1 = q_tru - q_nm1_bri1;
err_nm1_l2_bri1 = norm_sp(err_nm1_bri1)/q_tru_norm_sp;
errs_l2_bri1(1) = err_nm1_l2_bri1;

disp('BRI1 iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_bri1)])

for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_bri1 + f_frho(x,t,rho);
    [rho_n_bri1,phi_n_bri1,outs_n_bri1] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_bri1;
    opts.v_num = outs_n_bri1.v_num;

    phi0_n_bri1 = phi_n_bri1(:,1);
    gdphi_n_bri1 = op_grd(phi_n_bri1);
    term0_phi_n_bri1 = -vis*op_lap(phi0_n_bri1) + func_H(gdphi_n_bri1(:,1)) ...
                  + sum(phi0_n_bri1)*cst_int;

    % update q    
    res_phi_bri1 = term0_phi - term0_phi_n_bri1;
    q_n_bri1 = q_nm1_bri1 + res_phi_bri1;

    % record residue, error, number of iteration in ficplay
    nits_bri1(nit+1) = nits_bri1(nit) + length(outs_n_bri1.gaps_dual);
    gaps_bri1(nit) = outs_n_bri1.gaps_dual(end);
    ress_l2_bri1(nit) = norm_sp(phi0_n_bri1-phi0)/phi0_norm_sp;

    err_n_bri1 = q_tru - q_n_bri1;
    err_n_l2_bri1 = norm_sp(err_n_bri1)/q_tru_norm_sp;
    errs_l2_bri1(nit+1) = err_n_l2_bri1;

    % update index
    q_nm1_bri1 = q_n_bri1;
    % rho_nm1_bri1 = rho_n_bri1;
    err_nm1_bri1 = err_n_bri1;

    q_hist_bri1(:,nit) = q_n_bri1;
    
    if ress_l2_bri1(nit) < tol
        flag_cvg = true;
        q_hist_bri1 = q_hist_bri1(:,1:nit);
        nits_bri1 = nits_bri1(1:nit+1);
        gaps_bri1 = gaps_bri1(1:nit);
        ress_l2_bri1 = ress_l2_bri1(1:nit);
        errs_l2_bri1 = errs_l2_bri1(1:nit+1);
        disp(['BRI1 converges at ',num2str(nit),'-th iteration'])
        break
    end

end
if ~flag_cvg
    disp(['BRI1 finishes at ',num2str(nit),'-th iteration'])
end

%% inverse best response iteration (BRI, nit = 5)
flag_cvg = false;
opts.Nit = 5;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 100;
q_nm1_bri5 = q_init;
% rho_nm1_bri5 = rho_init;

q_hist_bri5 = zeros(nxp,Nit);
nits_bri5 = zeros(Nit+1,1);
gaps_bri5 = zeros(Nit,1);
errs_l2_bri5 = zeros(Nit+1,1);
ress_l2_bri5 = zeros(Nit,1);

err_nm1_bri5 = q_tru - q_nm1_bri5;
err_nm1_l2_bri5 = norm_sp(err_nm1_bri5)/q_tru_norm_sp;
errs_l2_bri5(1) = err_nm1_l2_bri5;

disp('BRI5 iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_bri5)])

for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_bri5 + f_frho(x,t,rho);
    [rho_n_bri5,phi_n_bri5,outs_n_bri5] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_bri5;
    opts.v_num = outs_n_bri5.v_num;

    phi0_n_bri5 = phi_n_bri5(:,1);
    gdphi_n_bri5 = op_grd(phi_n_bri5);
    term0_phi_n_bri5 = -vis*op_lap(phi0_n_bri5) + func_H(gdphi_n_bri5(:,1)) ...
                  + sum(phi0_n_bri5)*cst_int;

    % update q    
    res_phi_bri5 = term0_phi - term0_phi_n_bri5;
    q_n_bri5 = q_nm1_bri5 + res_phi_bri5;

    % record residue, error, number of iteration in ficplay
    nits_bri5(nit+1) = nits_bri5(nit) + length(outs_n_bri5.gaps_dual);
    gaps_bri5(nit) = outs_n_bri5.gaps_dual(end);
    ress_l2_bri5(nit) = norm_sp(phi0_n_bri5-phi0)/phi0_norm_sp;

    err_n_bri5 = q_tru - q_n_bri5;
    err_n_l2_bri5 = norm_sp(err_n_bri5)/q_tru_norm_sp;
    errs_l2_bri5(nit+1) = err_n_l2_bri5;


    % update index
    q_nm1_bri5 = q_n_bri5;
    % rho_nm1_bri5 = rho_n_bri5;
    err_nm1_bri5 = err_n_bri5;

    q_hist_bri5(:,nit) = q_n_bri5;

    if ress_l2_bri5(nit) < tol
        flag_cvg = true;
        q_hist_bri5 = q_hist_bri5(:,1:nit);
        nits_bri5 = nits_bri5(1:nit+1);
        gaps_bri5 = gaps_bri5(1:nit);
        ress_l2_bri5 = ress_l2_bri5(1:nit);
        errs_l2_bri5 = errs_l2_bri5(1:nit+1);
        disp(['BRI5 converges at ',num2str(nit),'-th iteration'])
        break
    end

end
if ~flag_cvg
    disp(['BRI5 finishes at ',num2str(nit),'-th iteration'])
end

%% save results
close all
save(['results/',filename])

%% show results
close all
% marker 'o', '+', 'x', '*', '.'
% linestyle '-', '--', ':', '-.'
% color "#0072BD" (blue), 	"#D95319" (orange), "#EDB120" (yellow)
% "#7E2F8E" (purple), "#77AC30" (green), 	"#A2142F" (dark red)
width = 7; height = 6;
color_bri1 = "#0072BD"; marker_bri1 = '+'; line_bri1 = '-.';
color_bri5 = "#D95319"; marker_bri5 = 'o'; line_bri5 = '--';
color_eci  = "#EDB120"; marker_eci  = 'x'; line_eci  = ':';
color_ref1 = "#7E2F8E"; 
color_ref2 = "#77AC30"; 

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(xgrid,q_tru,'Color',color_ref1,'DisplayName','true',...
    'LineWidth',linewidth);hold on
plot(xgrid,q_init,'Color',color_ref2,'DisplayName','initial',...
    'LineWidth',2*linewidth);
xlabel('x')
ylabel('$q(x)$',Interpreter='latex')
legend(Location='north')
exportgraphics(fig,['results/',filename,'_obs.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(3,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(xgrid,q_n_bri1-q_tru,...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
nexttile
plot(xgrid,q_n_bri5-q_tru,...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
nexttile
plot(xgrid,q_n_eci-q_tru,...
    'Color',color_eci,...%'Marker',marker_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel(fig, 'x')
ylabel(fig, '$\hat{q}(x) - q(x)$',Interpreter='latex')
exportgraphics(fig,['results/',filename,'_obserr.eps'],...
               'BackgroundColor','none');


fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(nits_bri1,errs_l2_bri1,...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
semilogy(nits_bri5,errs_l2_bri5,...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
semilogy(nits_eci,errs_l2_eci,...
    'Color',color_eci,'Marker',marker_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel('number of HJB/FP solved')
ylabel('environment rel. err.')
legend
exportgraphics(fig,['results/',filename,'_relerr_vsnbr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(errs_l2_bri1,...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
semilogy(errs_l2_bri5,...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
semilogy(errs_l2_eci,...
    'Color',color_eci,...%'Marker',marker_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('environment rel. err.')
legend
exportgraphics(fig,['results/',filename,'_relerr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(nits_bri1(2:end),ress_l2_bri1,...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
semilogy(nits_bri5(2:end),ress_l2_bri5,...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
semilogy(nits_eci(2:end),ress_l2_eci,...
    'Color',color_eci,'Marker',marker_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel('number of HJB/FP solved')
ylabel('measurement rel. err.')
legend()
exportgraphics(fig,['results/',filename,'_res_vsnbr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(ress_l2_bri1,...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
semilogy(ress_l2_bri5,...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
semilogy(ress_l2_eci,...
    'Color',color_eci,...%'Marker',marker_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('measurement rel. err.')
legend()
exportgraphics(fig,['results/',filename,'_res.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(abs(gaps_bri1),...
    'Color',color_bri1,...%'Marker',marker_bri1,...
    'LineStyle',line_bri1,'DisplayName','BRI1',...
    'LineWidth',linewidth);hold on
semilogy(abs(gaps_bri5),...
    'Color',color_bri5,...%'Marker',marker_bri5,...
    'LineStyle',line_bri5,'DisplayName','BRI5',...
    'LineWidth',linewidth);
semilogy(abs(gaps_eci),...
    'Color',color_eci,...
    'LineStyle',line_eci,'DisplayName','ECI',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('forward residue')
legend
exportgraphics(fig,['results/',filename,'_gapsdual.eps'],...
               'BackgroundColor','none');
