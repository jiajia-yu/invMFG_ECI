clear;
clc;
% close all;
        
fontsize = 20;
linewidth = 2;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
verbose = true;

%% domain
lend = 0; rend = 2; tend = 1;
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
vis = 0.1;
gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(p) 1/gamma * abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);

f_rho0 = @(x) gau(1,0.2,x); 
f_q = @(x) exp(sin(2*pi*x));
filename = 'bri_wrongimp'; 
f_frho = @(x,t,rho) rho;
f_g = @(x,rho)  zeros(size(x));

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
f_qinit = @(x) zeros(size(x));
rho_init = repmat(rho0,1,ntp);
q_init = f_qinit(xgrid);

%% inverse best response iteration (BRI, nit = 1, step = 0.5)
flag_cvg = false;

opts.Nit = 1;
opts.step = @(nit) 0.5;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 50;
q_nm1_bri1_ss05 = q_init;
% rho_nm1_bri1_ss05 = rho_init;

q_hist_bri1_ss05 = zeros(nxp,Nit);
nits_bri1_ss05 = zeros(Nit+1,1);
gaps_bri1_ss05 = zeros(Nit,1);
errs_l2_bri1_ss05 = zeros(Nit+1,1);
ress_l2_bri1_ss05 = zeros(Nit,1);

err_nm1_bri1_ss05 = q_tru - q_nm1_bri1_ss05;
err_nm1_l2_bri1_ss05 = norm_sp(err_nm1_bri1_ss05)/q_tru_norm_sp;
errs_l2_bri1_ss05(1) = err_nm1_l2_bri1_ss05;

disp('BRI1 iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_bri1_ss05)])

for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_bri1_ss05 + f_frho(x,t,rho);
    [rho_n_bri1_ss05,phi_n_bri1_ss05,outs_n_bri1_ss05] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_bri1_ss05;
    opts.v_num = outs_n_bri1_ss05.v_num;

    phi0_n_bri1_ss05 = phi_n_bri1_ss05(:,1);
    gdphi_n_bri1_ss05 = op_grd(phi_n_bri1_ss05);
    term0_phi_n_bri1_ss05 = -vis*op_lap(phi0_n_bri1_ss05) + func_H(gdphi_n_bri1_ss05(:,1)) ...
                  + sum(phi0_n_bri1_ss05)*cst_int;

    % update q    
    res_phi_bri1_ss05 = term0_phi - term0_phi_n_bri1_ss05;
    q_n_bri1_ss05 = q_nm1_bri1_ss05 + res_phi_bri1_ss05;

    % record residue, error, number of iteration in ficplay
    nits_bri1_ss05(nit+1) = nits_bri1_ss05(nit) + length(outs_n_bri1_ss05.gaps_dual);
    gaps_bri1_ss05(nit) = outs_n_bri1_ss05.gaps_dual(end);
    ress_l2_bri1_ss05(nit) = norm_sp(phi0_n_bri1_ss05-phi0)/phi0_norm_sp;

    err_n_bri1_ss05 = q_tru - q_n_bri1_ss05;
    err_n_l2_bri1_ss05 = norm_sp(err_n_bri1_ss05)/q_tru_norm_sp;
    errs_l2_bri1_ss05(nit+1) = err_n_l2_bri1_ss05;

    % update index
    q_nm1_bri1_ss05 = q_n_bri1_ss05;
    % rho_nm1_bri1_ss05 = rho_n_bri1_ss05;
    err_nm1_bri1_ss05 = err_n_bri1_ss05;

    q_hist_bri1_ss05(:,nit) = q_n_bri1_ss05;
    
    if ress_l2_bri1_ss05(nit) < tol
        flag_cvg = true;
        q_hist_bri1_ss05 = q_hist_bri1_ss05(:,1:nit);
        nits_bri1_ss05 = nits_bri1_ss05(1:nit+1);
        gaps_bri1_ss05 = gaps_bri1_ss05(1:nit);
        ress_l2_bri1_ss05 = ress_l2_bri1_ss05(1:nit);
        errs_l2_bri1_ss05 = errs_l2_bri1_ss05(1:nit+1);
        disp(['BRI1 ss0.5 converges at ',num2str(nit),'-th iteration'])
        break
    end

end
if ~flag_cvg
    disp(['BRI1 ss0.5 finishes at ',num2str(nit),'-th iteration'])
end

%% inverse best response iteration (BRI, nit = 1, step = 1)
flag_cvg = false;

opts.Nit = 1;
opts.step = @(nit) 1;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 50;
q_nm1_bri1_ss1 = q_init;
% rho_nm1_bri1_ss1 = rho_init;

q_hist_bri1_ss1 = zeros(nxp,Nit);
nits_bri1_ss1 = zeros(Nit+1,1);
gaps_bri1_ss1 = zeros(Nit,1);
errs_l2_bri1_ss1 = zeros(Nit+1,1);
ress_l2_bri1_ss1 = zeros(Nit,1);

err_nm1_bri1_ss1 = q_tru - q_nm1_bri1_ss1;
err_nm1_l2_bri1_ss1 = norm_sp(err_nm1_bri1_ss1)/q_tru_norm_sp;
errs_l2_bri1_ss1(1) = err_nm1_l2_bri1_ss1;

disp('BRI1 iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_bri1_ss1)])

for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_bri1_ss1 + f_frho(x,t,rho);
    [rho_n_bri1_ss1,phi_n_bri1_ss1,outs_n_bri1_ss1] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_bri1_ss1;
    opts.v_num = outs_n_bri1_ss1.v_num;

    phi0_n_bri1_ss1 = phi_n_bri1_ss1(:,1);
    gdphi_n_bri1_ss1 = op_grd(phi_n_bri1_ss1);
    term0_phi_n_bri1_ss1 = -vis*op_lap(phi0_n_bri1_ss1) + func_H(gdphi_n_bri1_ss1(:,1)) ...
                  + sum(phi0_n_bri1_ss1)*cst_int;

    % update q    
    res_phi_bri1_ss1 = term0_phi - term0_phi_n_bri1_ss1;
    q_n_bri1_ss1 = q_nm1_bri1_ss1 + res_phi_bri1_ss1;

    % record residue, error, number of iteration in ficplay
    nits_bri1_ss1(nit+1) = nits_bri1_ss1(nit) + length(outs_n_bri1_ss1.gaps_dual);
    gaps_bri1_ss1(nit) = outs_n_bri1_ss1.gaps_dual(end);
    ress_l2_bri1_ss1(nit) = norm_sp(phi0_n_bri1_ss1-phi0)/phi0_norm_sp;

    err_n_bri1_ss1 = q_tru - q_n_bri1_ss1;
    err_n_l2_bri1_ss1 = norm_sp(err_n_bri1_ss1)/q_tru_norm_sp;
    errs_l2_bri1_ss1(nit+1) = err_n_l2_bri1_ss1;

    % update index
    q_nm1_bri1_ss1 = q_n_bri1_ss1;
    % rho_nm1_bri1_ss1 = rho_n_bri1_ss1;
    err_nm1_bri1_ss1 = err_n_bri1_ss1;

    q_hist_bri1_ss1(:,nit) = q_n_bri1_ss1;
    
    if ress_l2_bri1_ss1(nit) < tol
        flag_cvg = true;
        q_hist_bri1_ss1 = q_hist_bri1_ss1(:,1:nit);
        nits_bri1_ss1 = nits_bri1_ss1(1:nit+1);
        gaps_bri1_ss1 = gaps_bri1_ss1(1:nit);
        ress_l2_bri1_ss1 = ress_l2_bri1_ss1(1:nit);
        errs_l2_bri1_ss1 = errs_l2_bri1_ss1(1:nit+1);
        disp(['BRI1 ss1 converges at ',num2str(nit),'-th iteration'])
        break
    end

end
if ~flag_cvg
    disp(['BRI1 ss1 finishes at ',num2str(nit),'-th iteration'])
end


%% inverse best response iteration (BRI, nit = 5)
flag_cvg = false;
opts.Nit = 1;
opts.step = @(nit) 0.5;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 50;
q_nm1_rst_ss5 = q_init;
% rho_nm1_rst_ss5 = rho_init;

q_hist_rst_ss5 = zeros(nxp,Nit);
nits_rst_ss5 = zeros(Nit+1,1);
gaps_rst_ss5 = zeros(Nit,1);
errs_l2_rst_ss5 = zeros(Nit+1,1);
ress_l2_rst_ss5 = zeros(Nit,1);

err_nm1_rst_ss5 = q_tru - q_nm1_rst_ss5;
err_nm1_l2_rst_ss5 = norm_sp(err_nm1_rst_ss5)/q_tru_norm_sp;
errs_l2_rst_ss5(1) = err_nm1_l2_rst_ss5;

disp('rst_ss5 iter 0')
disp(['  initial: q rel err = ',num2str(err_nm1_l2_rst_ss5)])

for nit = 1:Nit
    % update rho, phi through ficplay
    f_f = @(x,t,rho) q_nm1_rst_ss5 + f_frho(x,t,rho);
    [rho_n_rst_ss5,phi_n_rst_ss5,outs_n_rst_ss5] ...
    = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    % opts.rho_num = rho_n_rst_ss5;
    % opts.v_num = outs_n_rst_ss5.v_num;

    phi0_n_rst_ss5 = phi_n_rst_ss5(:,1);
    gdphi_n_rst_ss5 = op_grd(phi_n_rst_ss5);
    term0_phi_n_rst_ss5 = -vis*op_lap(phi0_n_rst_ss5) + func_H(gdphi_n_rst_ss5(:,1)) ...
                  + sum(phi0_n_rst_ss5)*cst_int;

    % update q    
    res_phi_rst_ss5 = term0_phi - term0_phi_n_rst_ss5;
    q_n_rst_ss5 = q_nm1_rst_ss5 + res_phi_rst_ss5;

    % record residue, error, number of iteration in ficplay
    nits_rst_ss5(nit+1) = nits_rst_ss5(nit) + length(outs_n_rst_ss5.gaps_dual);
    gaps_rst_ss5(nit) = outs_n_rst_ss5.gaps_dual(end);
    ress_l2_rst_ss5(nit) = norm_sp(phi0_n_rst_ss5-phi0)/phi0_norm_sp;

    err_n_rst_ss5 = q_tru - q_n_rst_ss5;
    err_n_l2_rst_ss5 = norm_sp(err_n_rst_ss5)/q_tru_norm_sp;
    errs_l2_rst_ss5(nit+1) = err_n_l2_rst_ss5;


    % update index
    q_nm1_rst_ss5 = q_n_rst_ss5;
    % rho_nm1_rst_ss5 = rho_n_rst_ss5;
    err_nm1_rst_ss5 = err_n_rst_ss5;

    q_hist_rst_ss5(:,nit) = q_n_rst_ss5;

    if ress_l2_rst_ss5(nit) < tol
        flag_cvg = true;
        q_hist_rst_ss5 = q_hist_rst_ss5(:,1:nit);
        nits_rst_ss5 = nits_rst_ss5(1:nit+1);
        gaps_rst_ss5 = gaps_rst_ss5(1:nit);
        ress_l2_rst_ss5 = ress_l2_rst_ss5(1:nit);
        errs_l2_rst_ss5 = errs_l2_rst_ss5(1:nit+1);
        disp(['rst_ss5 converges at ',num2str(nit),'-th iteration'])
        break
    end

end
if ~flag_cvg
    disp(['rst_ss5 finishes at ',num2str(nit),'-th iteration'])
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
width = 8; height = 6; fontsize = 12;
color_bri1_ss1 = "#0072BD"; marker_bri1_ss1 = '+'; line_bri1_ss1 = '-';
color_bri1_ss05= "#A2142F"; marker_bri1_ss05= '*'; line_bri1_ss05= ':';
color_rst_ss5 = "#EDB120"; marker_rst_ss5 = 'o'; line_rst_ss5 = '-.';
% color_eci  = "#77AC30"; marker_eci  = 'x'; line_eci  = '-';
color_ref1 = "#7E2F8E"; 
% color_ref2 = "#77AC30"; 

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(xgrid,q_tru,'Color',color_ref1,'DisplayName','true',...
    'LineWidth',linewidth,'LineStyle','--');hold on
xlabel('$x$',Interpreter='latex')
ylabel('$q(x)$',Interpreter='latex')
legend(Location='northeast')
grid on
exportgraphics(fig,['results/',filename,'_obs.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(xgrid,q_n_bri1_ss05-q_tru,...
    'Color',color_bri1_ss05,...%'Marker',marker_bri,...
    'LineStyle',line_bri1_ss05,'DisplayName','BRI(0.5)',...
    'LineWidth',linewidth);hold on
plot(xgrid,q_n_bri1_ss1-q_tru,...
    'Color',color_bri1_ss1,...%'Marker',marker_bri1_ss1,...
    'LineStyle',line_bri1_ss1,'DisplayName','BRI(1)',...
    'LineWidth',linewidth);
plot(xgrid,q_n_rst_ss5-q_tru,...
    'Color',color_rst_ss5,...%'Marker',marker_rst_ss5,...
    'LineStyle',line_rst_ss5,'DisplayName','RST(0.5)',...
    'LineWidth',linewidth);
xlabel(fig, '$x$',Interpreter='latex')
ylabel(fig, '$\hat{q}(x) - q(x)$',Interpreter='latex')
legend(Location="northwest")
grid on
exportgraphics(fig,['results/',filename,'_obserr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(errs_l2_bri1_ss05,...
    'Color',color_bri1_ss05,...%'Marker',marker_bri,...
    'LineStyle',line_bri1_ss05,'DisplayName','BRI1(0.5)',...
    'LineWidth',linewidth);hold on
semilogy(errs_l2_bri1_ss1,...
    'Color',color_bri1_ss1,...%'Marker',marker_bri1_ss1,...
    'LineStyle',line_bri1_ss1,'DisplayName','BRI1(1)',...
    'LineWidth',linewidth);hold on
semilogy(errs_l2_rst_ss5,...
    'Color',color_rst_ss5,...%'Marker',marker_rst_ss5,...
    'LineStyle',line_rst_ss5,'DisplayName','RST(0.5)',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('$q$ rel. err.',Interpreter='latex',FontSize=fontsize)
legend(Location="southwest")
grid on
exportgraphics(fig,['results/',filename,'_relerr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(ress_l2_bri1_ss05,...
    'Color',color_bri1_ss05,...%'Marker',marker_bri,...
    'LineStyle',line_bri1_ss05,'DisplayName','BRI1(0.5)',...
    'LineWidth',linewidth);hold on
semilogy(ress_l2_bri1_ss1,...
    'Color',color_bri1_ss1,...%'Marker',marker_bri1_ss1,...
    'LineStyle',line_bri1_ss1,'DisplayName','BRI1(1)',...
    'LineWidth',linewidth);hold on
semilogy(ress_l2_rst_ss5,...
    'Color',color_rst_ss5,...%'Marker',marker_rst_ss5,...
    'LineStyle',line_rst_ss5,'DisplayName','RST(0.5)',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('$\phi_0$ rel. err.',Interpreter='latex',FontSize=fontsize)
legend(Location="southwest")
grid on
exportgraphics(fig,['results/',filename,'_res.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(abs(gaps_bri1_ss05),...
    'Color',color_bri1_ss05,...%'Marker',marker_bri,...
    'LineStyle',line_bri1_ss05,'DisplayName','BRI1(0.5)',...
    'LineWidth',linewidth);hold on
semilogy(abs(gaps_bri1_ss1),...
    'Color',color_bri1_ss1,...%'Marker',marker_bri1_ss1,...
    'LineStyle',line_bri1_ss1,'DisplayName','BRI1(1)',...
    'LineWidth',linewidth);hold on
semilogy(abs(gaps_rst_ss5),...
    'Color',color_rst_ss5,...%'Marker',marker_rst_ss5,...
    'LineStyle',line_rst_ss5,'DisplayName','RST(0.5)',...
    'LineWidth',linewidth);
xlabel('number of outer loop')
ylabel('forward residue')
legend(Location="southwest")
grid on
exportgraphics(fig,['results/',filename,'_gapsdual.eps'],...
               'BackgroundColor','none');
