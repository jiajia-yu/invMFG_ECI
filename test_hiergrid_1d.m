clear;
clc;
% close all;
        
fontsize = 20;
linewidth = 2;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
verbose = true;

casename = 'test1';
f_q = @(x) exp(x).*(sin(2*pi*x));
% casename = 'test2';
% f_q = @(x) exp(x).*(sin(8*pi*x));
% casename = 'test3';
% f_q = @(x) exp(x).*sin(2*pi*(x+1).^2);

filename = ['hiergrid_',casename]; 

%% domain
lend = -1; rend = 1; tend = 1;
nx = 2^10; nt = 2^9;
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

% for hier grid
L = 4;
list_nx = zeros(L,1);
list_nt = zeros(L,1);
nx_l = nx; nt_l = nt;
for l = L:-1:1
    list_nx(l) = nx_l;
    list_nt(l) = nt_l;

    nx_l = nx_l/2; nt_l = nt_l/2;
end

%% true
% gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
vis = 0.1;
gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(p) 1/gamma * abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);

f_rho0 = @(x) 1/(rend-lend)*ones(size(x));% gau(0,0.2,x); 
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

%% forward problem (one grid)
% q_tru = f_q(xgrid); %q_tru = q_tru - min(q_tru);
% q_tru_norm_sp = norm_sp(q_tru);
% f_f = @(x,t,rho) q_tru + f_frho(x,t,rho);
% rho0 = f_rho0(xgrid);
% 
% tic
% [rho_tru,phi_tru,outs_tru] ...
%     = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
% t_run = toc;
% res_tru = outs_tru.gaps_dual(end);
% 
% disp(['forward solver residue = ', num2str(res_tru)]);
% 
% figure(4);clf;
% set(gcf,'unit','centimeters','Position',[3 30 24 7]);
% subplot(131);plot(xgrid,q_tru,'LineWidth',linewidth);title('true obstacle')
% subplot(132);mesh(xmesh,tmesh,phi_tru);xlabel('x');ylabel('t');title('true phi')
% subplot(133);mesh(xmesh,tmesh,rho_tru);xlabel('x');ylabel('t');title('true rho')


%% forward problem (hier grid)
for l = 1:L
    % set up
    disp(['layer ',num2str(l)]);
    nx_l = list_nx(l); nt_l = list_nt(l);
    xgrid_l = linspace(lend,rend,nx_l+1)';
    q_tru_l = f_q(xgrid_l); 
    
    opts.nx = nx_l; opts.nt = nt_l;
    f_f = @(x,t,rho) q_tru_l + f_frho(x,t,rho);
    % solve
    tic
    [rho_tru_l,phi_tru_l,outs_tru_l] ...
        = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
    res_tru = outs_tru_l.gaps_dual(end);
    % display residue
    disp(['forward solver residue = ', num2str(res_tru)]);
    % refine mesh
    if l < L
        opts.rho_num = mg1d_interp(rho_tru_l);
        opts.v_num = mg1d_interp(outs_tru_l.v_num);
    end
end
x_grid = xgrid_l; q_tru = q_tru_l; q_tru_norm_sp = norm_sp(q_tru);
rho0 = f_rho0(xgrid);
rho_tru = rho_tru_l; phi_tru = phi_tru_l; outs_tru = outs_tru_l;

figure(4);clf;
set(gcf,'unit','centimeters','Position',[3 30 24 7]);
subplot(131);plot(xgrid,q_tru,'LineWidth',linewidth);title('true obstacle')
subplot(132);mesh(xmesh,tmesh,phi_tru);xlabel('x');ylabel('t');title('true phi')
subplot(133);mesh(xmesh,tmesh,rho_tru);xlabel('x');ylabel('t');title('true rho')


%% measurement, potentially noisy
phi0 = phi_tru(:,1); phi0_norm_sp = norm_sp(phi0);
term0_phi = -vis*op_lap(phi0) + op_ham(phi0);
dtphi0 = term0_phi - q_tru - f_frho(xgrid,0,rho0);

cst_int = dx/tend/(rend-lend);
term0_phi = term0_phi  + sum(phi0)*cst_int;

% for hier grid
cell_xgrid = cell(L,1);
cell_term0_phi = cell(L,1);
cell_phi0 = cell(L,1);
cell_qtru = cell(L,1);
xgrid_l = xgrid;
term0_phi_l = term0_phi;
phi0_l = phi0;
q_tru_l = q_tru;
for l = L:-1:1
    cell_xgrid{l} = xgrid_l;
    cell_term0_phi{l} = term0_phi_l;
    cell_phi0{l} = phi0_l;
    cell_qtru{l} = q_tru_l;

    xgrid_l = xgrid_l(1:2:end);
    term0_phi_l = mg1d_restrc_sp(term0_phi_l);
    phi0_l = mg1d_restrc_sp(phi0_l);
    q_tru_l = mg1d_restrc_sp(q_tru_l);
end

%% inverse general setting
% HJB parameter
opts_hjb = opts;
if isfield(opts,'hjb_verbose') opts_hjb.verbose = opts.hjb_verbose; end
% FP parameter
opts_fp = opts;
if isfield(opts,'fp_verbose') opts_fp.verbose = opts.fp_verbose; end

tol = 1e-6;
f_qinit = @(x) zeros(size(x));
rho_init = repmat(rho0,1,ntp);
q_init = f_qinit(xgrid);

%% one grid
flag_cvg = false;

opts.Nit = 100;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 100;
q_nm1_eci = q_init;
% rho_nm1_eci = rho_init;

q_hist_eci = zeros(nxp,Nit);
nits_eci = zeros(Nit+1,1);
errs_l2_eci = zeros(Nit+1,1);
ress_l2_eci = zeros(Nit,1);

err_nm1_eci = q_tru - q_nm1_eci;
err_nm1_l2_eci = norm_sp(err_nm1_eci)/q_tru_norm_sp;
errs_l2_eci(1) = err_nm1_l2_eci;

disp('one grid iter 0')
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
    nits_eci(nit+1) = nits_eci(nit) + length(outs_n_eci.gaps_dual)+1;
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
        ress_l2_eci = ress_l2_eci(1:nit);
        errs_l2_eci = errs_l2_eci(1:nit+1);
        disp(['one grid converges at ',num2str(nit),'-th iteration'])
        break
    end

end
t_eci = toc;

if ~flag_cvg
    disp(['one grid finishes at ',num2str(nit),'-th iteration'])
end

disp(['one grid total time: ', num2str(t_eci),'seconds'])

%% hier grid
% remove previous result
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'v_num')opts = rmfield(opts,'v_num'); end

Nit = 100;

% set up recorders
list_time = zeros(L,1);
cell_q_hist = cell(L,1);
cell_nit = cell(L,1);
cell_gaps = cell(L,1);
cell_errs = cell(L,1);
cell_ress = cell(L,1);


% initialize q
xgrid_l = linspace(lend,rend,list_nx(1)+1)';
q_init = f_qinit(xgrid_l);

for l = 1:L
    % set up grids
    nx_l = list_nx(l); nt_l = list_nt(l);
    dx_l = (rend-lend)/nx_l; dt_l = tend/nt_l;
    opts.nx = nx_l; opts.nt = nt_l;
    cst_int_l = dx_l/tend/(rend-lend);

    op_lap_l = @(u) cat(1, u(end-1,  :) - 2*u(1,      :) + u(2,    :), ...
                         u(1:end-2,:) - 2*u(2:end-1,:) + u(3:end,:), ...
                         u(  end-1,:) - 2*u(  end,  :) + u(  2,  :) )/(dx_l^2); 
    op_grd_l = @(u) cat(1, u(2,    :) - u(end-1,  :), ...
                         u(3:end,:) - u(1:end-2,:), ...
                         u(  2,  :) - u(  end-1,:) )/(2*dx_l);
    op_ham_l = @(u) func_H(op_grd_l(u));
    
    norm_sp_l = @(u) sqrt(sum(u.^2,'all')*dx_l);
    norm_spt_l = @(u) sqrt(sum(u.^2,'all')*dx_l*dt_l);

    % set up recorders
    q_hist_l = zeros(nx_l+1,Nit);
    nits_l = zeros(Nit+1,1);
    gaps_l = zeros(Nit,1);
    errs_l2_l = zeros(Nit+1,1);
    ress_l2_l = zeros(Nit,1);

    % set up grids and observation
    term0_phi_l = cell_term0_phi{l};
    phi0_l = cell_phi0{l};
    q_tru_l = cell_qtru{l};

    % set up initialization
    q_nm1_l = q_init;
    err_nm1_l = q_tru_l - q_nm1_l;
    err_nm1_l2_l = norm_sp_l(err_nm1_l)/q_tru_norm_sp;
    errs_l2_l(1) = err_nm1_l2_l;
    
    disp(['hier grid layer-', num2str(l), ' iter 0'])
    disp(['  initial: q rel err = ',num2str(err_nm1_l2_l)])

    % run solver on the l-th grid
    flag_cvg = false;
    tic
    for nit = 1:Nit
        % update rho, phi through ficplay
        f_f = @(x,t,rho) q_nm1_l + f_frho(x,t,rho);
        [rho_n_l,phi_n_l,outs_n_l] ...
        = ficplay1d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
    
        opts.rho_num = rho_n_l;
        opts.v_num = outs_n_l.v_num;
    
        phi0_n_l = phi_n_l(:,1);
        gdphi_n_l = op_grd_l(phi_n_l);
        term0_phi_n_l = -vis*op_lap_l(phi0_n_l) + func_H(gdphi_n_l(:,1)) ...
                      + sum(phi0_n_l)*cst_int_l;
    
        % update q    
        res_phi_l = term0_phi_l - term0_phi_n_l;
        q_n_l = q_nm1_l + res_phi_l;
    
        % record residue, error, number of iteration in ficplay
        nits_l(nit+1) = nits_l(nit) + length(outs_n_l.gaps_dual);
        gaps_l(nit) = outs_n_l.gaps_dual(end);
        % ress_l2_l(nit) = norm_sp_l(phi0_n_l-phi0_l)/phi0_norm_sp;
        ress_l2_l(nit) = norm_sp_l(res_phi_l);

        err_n_l = q_tru_l - q_n_l;
        err_n_l2_l = norm_sp_l(err_n_l)/q_tru_norm_sp;
        errs_l2_l(nit+1) = err_n_l2_l;
    
        % update index
        q_nm1_l = q_n_l;
        % rho_nm1_bri1 = rho_n_bri1;
        err_nm1_l = err_n_l;
    
        q_hist_l(:,nit) = q_n_l;
        
        if ress_l2_l(nit) < tol
            flag_cvg = true;
            q_hist_l = q_hist_l(:,1:nit);
            nits_l = nits_l(1:nit+1);
            gaps_l = gaps_l(1:nit);
            ress_l2_l = ress_l2_l(1:nit);
            errs_l2_l = errs_l2_l(1:nit+1);
            disp(['current layer converges at ',num2str(nit),'-th iteration'])
            break
        end
    
    end
    if ~flag_cvg
        disp(['current layer finishes at ',num2str(nit),'-th iteration'])
    end

    % record results and output
    list_time(l) = toc;
    cell_q_hist{l} = q_hist_l;
    cell_nit{l} = nits_l;
    cell_gaps{l} = gaps_l;
    cell_errs{l} = errs_l2_l;
    cell_ress{l} = ress_l2_l;

    if l < L
        % refine mesh
        q_init = mg1d_interp_sp(linspace(lend,rend,list_nx(l)+1)',...
                        q_nm1_l,linspace(lend,rend,list_nx(l+1)+1)');
        opts.rho_num = mg1d_interp(rho_n_l);
        opts.v_num = mg1d_interp(outs_n_l.v_num);
    end
end

%% organize results
times_eci = nits_eci/nits_eci(end) * t_eci;

nit0 = 0; nits_hier = [0];
t0 = 0; times_hier = [0];
errs_l2_hier = [cell_errs{1}(1)];
ress_l2_hier = [];
for l = 1:L
    nits_hier = [nits_hier; nit0 + cell_nit{l}(2:end)];
    times_hier = [times_hier; 
                  t0 + cell_nit{l}(2:end)/cell_nit{l}(end) * list_time(l)];
    errs_l2_hier = [errs_l2_hier; cell_errs{l}(2:end)];
    ress_l2_hier = [ress_l2_hier; cell_ress{l}];
    nit0 = nit0 + cell_nit{l}(end);
    t0 = t0 + list_time(l);
end

close all
save(['results/',filename])

%% show results
color_1 = "#0072BD"; marker_1 = '+'; line_1 = '-.';
color_2 = "#D95319"; marker_2 = 'o'; line_2 = ':';
color_3 = "#EDB120"; marker_3 = 'x'; line_3 = '--';
color_ref1 = "#7E2F8E"; 
color_ref2 = "#77AC30"; 

width = 14; height = 5;

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
semilogy(times_eci,errs_l2_eci,'-.',...
    'LineWidth',linewidth,'DisplayName','ECI');hold on
semilogy(times_hier,errs_l2_hier,'-',...
    'LineWidth',linewidth,'DisplayName','HECI');
xlabel('time elapsed (sec)')
ylabel('relative error')
legend
exportgraphics(fig,['results/',filename,'_relerr_vstime.eps'],...
               'BackgroundColor','none');

width = 7; height = 6;

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(xgrid,q_tru,'Color',color_ref1,...
    'LineWidth',linewidth,'DisplayName','ECI');
xlabel('x')
ylabel('q(x)')
exportgraphics(fig,['results/',filename,'_obs.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
nexttile
plot(cell_xgrid{1},cell_qtru{1}-cell_q_hist{1}(:,end),...
    'LineStyle',line_1,'Color',color_1,...
    'LineWidth',linewidth,'DisplayName','level1'); hold on
plot(cell_xgrid{2},cell_qtru{2}-cell_q_hist{2}(:,end),...
    'LineStyle',line_2,'Color',color_2,...
    'LineWidth',linewidth,'DisplayName','level2');
plot(cell_xgrid{3},cell_qtru{3}-cell_q_hist{3}(:,end),...
    'LineStyle',line_3,'Color',color_3,...
    'LineWidth',linewidth,'DisplayName','level3');
plot(cell_xgrid{4},cell_qtru{4}-cell_q_hist{4}(:,end),...
    'Color',color_ref2,...
    'LineWidth',linewidth,'DisplayName','level4');
xlabel('x')
legend(Location='southwest')
title('$q(x)-\hat{q}(x)$','Interpreter','latex')
exportgraphics(fig,['results/',filename,'_obserr.eps'],...
               'BackgroundColor','none');


% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[7 7 width height]);% >,^,
% 
% nexttile
% semilogy(nits_eci,errs_l2_eci,'-.',...
%     'LineWidth',linewidth,'DisplayName','ECI');hold on
% semilogy(nits_hier,errs_l2_hier,'-',...
%     'LineWidth',linewidth,'DisplayName','HECI');
% xlabel('number of HJB/FP solved')
% ylabel('l2 relative error of obstacle')
% legend
% exportgraphics(fig,['results/',filename,'_relerr_vsnit.eps'],...
%                'BackgroundColor','none');
% 
% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[21 7 width height]);
% 
% nexttile
% semilogy(nits_eci(2:end),ress_l2_eci,'-.',...
%     'LineWidth',linewidth,'DisplayName','ECI');hold on
% semilogy(nits_hier(2:end),ress_l2_hier,'-',...
%     'LineWidth',linewidth,'DisplayName','HECI');
% xlabel('number of HJB/FP solved')
% ylabel('consecutive residue')
% legend
% exportgraphics(fig,['results/',filename,'_res_vsnit.eps'],...
%                'BackgroundColor','none');

% fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
% set(gcf,'unit','centimeters','Position',[21 7 width height]);
% 
% nexttile
% semilogy(times_eci(2:end),ress_l2_eci,'-.',...
%     'LineWidth',linewidth,'DisplayName','ECI');hold on
% semilogy(times_hier(2:end),ress_l2_hier,'-',...
%     'LineWidth',linewidth,'DisplayName','HECI');
% xlabel('time elapsed (sec)')
% ylabel('consecutive residue')
% legend
% exportgraphics(fig,['results/',filename,'_res_vstime.eps'],...
%                'BackgroundColor','none');

