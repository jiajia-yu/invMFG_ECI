clear;
clc;
% close all;
        
fontsize = 20;
linewidth = 2;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));

%% domain
lend = 0; rend = 1; tend = 1;
nx = 2000; nt = 500;
opts = [];
opts.xrange = [lend,rend];
opts.tend = tend;
opts.nx = nx;
opts.nt = nt;
opts.vis_num = 1;
opts.bd = 'periodic';

opts.Nit = 100;
opts.step = @(nit) 0.5;
opts.back_track = false;
opts.tol = 1e-8;
opts.verbose = true;
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
verbose = true;

vis = 0.2; filename = '1dillu_vis2e-1';
gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(p) 1/gamma * abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);

f_rho0 = @(x) gau(0.5,0.1,x);
% Define multiscale + discontinuous potential function
f_q = @(x) ...
    (x < 0.4 | x > 0.7) .* ( sin(20*pi*x) .* exp(-10*(x - 0.5).^2) ) + ...
    (x > 0.4 & x < 0.7) .* (-exp(x)) + ...                           
    (x > 0.7) .* (0.2*sin(100*pi*x)) + ...           
    (x > 0.3 & x < 0.35) * (-1) + (x > 0.6 & x < 0.65) * (1);
% f_rho0 = @(x) (f_q(x)-min(f_q(x)))/sum(f_q(x)-min(f_q(x)))/dx; 
% f_frho = @(x,t,rho) rho; filename = [filename,'_local'];
f_g = @(x,rho)  zeros(size(rho));

ker_K = gau(0.5,0.1,xgrid); 
f_frho = @(x,t,rho) 1e0*dx*convn(rho.^2,ker_K,'same'); 
filename = [filename,'_nonlocal'];

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
subplot(231);plot(xgrid,q_tru,'LineWidth',linewidth);title('true obstacle')
subplot(232);mesh(xmesh,tmesh,phi_tru);xlabel('x');ylabel('t');title('true phi')
subplot(233);mesh(xmesh,tmesh,rho_tru);xlabel('x');ylabel('t');title('true rho')


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
f_qinit = @(x) rand(size(x))*(max(q_tru)-min(q_tru))+min(q_tru);
rho_init = repmat(rho0,1,ntp);
q_init = f_qinit(xgrid); q_init(1) = 0; q_init(end) = 0;

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

%% save results
close all
save(['results/',filename])

%% show results
close all
% marker 'o', '+', 'x', '*', '.'
% linestyle '-', '--', ':', '-.'
% color "#0072BD" (blue), 	"#D95319" (orange), "#EDB120" (yellow)
% "#7E2F8E" (purple), "#77AC30" (green), 	"#A2142F" (dark red)
img_width = 10; img_height = 4; fontsize = 14;
color_eci  = "#EDB120"; marker_eci  = 'x'; line_eci  = '-';
color_ref1 = "#7E2F8E"; 
color_ref2 = "#77AC30"; 
color_ref3 = "#0072BD";

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
semilogy(errs_l2_eci,'Color',color_ref3,...
    'LineWidth',linewidth);hold on
xlabel('number of outer loop')
ylabel('$q$ rel. err.',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_relerr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
semilogy(ress_l2_eci,'Color',color_ref3,...
    'LineWidth',linewidth);hold on
xlabel('number of outer loop')
ylabel('$\phi_0$ rel. err.',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_res.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
plot(xgrid,q_tru,'Color',color_ref1,'LineWidth',linewidth);hold on
xlabel('$x$',Interpreter='latex',FontSize=fontsize)
ylabel('$q(x)$',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_obstru.eps'],...
               'BackgroundColor','none');

% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
% nexttile
% plot(xgrid,q_init,'Color',color_eci,'LineWidth',linewidth);hold on
% xlabel('$x$',Interpreter='latex',FontSize=fontsize)
% ylabel('$q^{(0)}(x)$',Interpreter='latex',FontSize=fontsize)
% grid on
% exportgraphics(fig,['results/',filename,'_obsnum0.eps'],...
%                'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
plot(xgrid,q_hist_eci(:,1),'Color',color_eci,'LineWidth',linewidth);hold on
xlabel('$x$',Interpreter='latex',FontSize=fontsize)
ylabel('$q^{(1)}(x)$',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_obsnum1.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
plot(xgrid,q_n_eci,'Color',color_eci,'LineWidth',linewidth);hold on
xlabel('$x$',Interpreter='latex',FontSize=fontsize)
ylabel('$q^{(K)}(x)$',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_obsnum.eps'],...
               'BackgroundColor','none');

% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
% nexttile
% plot(xgrid,q_init-q_tru,'Color',color_ref2,'LineWidth',linewidth);hold on
% xlabel('$x$',Interpreter='latex',FontSize=fontsize)
% ylabel('$q^{(0)}(x)-q(x)$',Interpreter='latex',FontSize=fontsize)
% exportgraphics(fig,['results/',filename,'_obserr0.eps'],...
%                'BackgroundColor','none');
% 
% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
% nexttile
% plot(xgrid,q_hist_eci(:,1)-q_tru,'Color',color_eci,'LineWidth',linewidth);hold on
% xlabel('$x$',Interpreter='latex',FontSize=fontsize)
% ylabel('$q^{(1)}(x)-q(x)$',Interpreter='latex',FontSize=fontsize)
% exportgraphics(fig,['results/',filename,'_obserr1.eps'],...
%                'BackgroundColor','none');
% 
% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
% nexttile
% plot(xgrid,q_n_eci-q_tru,'Color',color_eci,'LineWidth',linewidth);hold on
% xlabel('$x$',Interpreter='latex',FontSize=fontsize)
% ylabel('$q^{(K)}(x)-q(x)$',Interpreter='latex',FontSize=fontsize)
% exportgraphics(fig,['results/',filename,'_obserr.eps'],...
%                'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
plot(xgrid,term0_phi,'Color',color_ref2,'LineWidth',linewidth);hold on
xlabel('$x$',Interpreter='latex',FontSize=fontsize)
ylabel('$M(\phi_0)$',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_term_phi0.eps'],...
               'BackgroundColor','none');
