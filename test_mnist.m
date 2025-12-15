% img = imread('mnist_grid.png');
% % img = rgb2gray(img);
% q_tru = double(img)/255;
% imshow(q_tru,[])
% save('mnist_grid.mat',"q_tru")
%%
clear
clc
close all

load("mnist_grid.mat")
% make q_tru periodic
% q_tru = padarray(q_tru,[1,1],0,'both');
[m,n] = size(q_tru);
q_tru_pad = zeros(m+2, n+2);
q_tru_pad(2:end-1, 2:end-1) = q_tru;
q_tru = q_tru_pad;
clear q_tru_pad

fontsize = 20;
linewidth = 2;
gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
verbose = true;

%% domain
lend = 0; rend = 1; tend = 1;
nx = size(q_tru,1)-1; ny = size(q_tru,2)-1; nt = 50;
opts = [];
opts.xrange = [lend,rend];
opts.yrange = [lend,rend];
opts.tend = tend;
opts.nx = nx;
opts.ny = ny;
opts.nt = nt;
opts.vis_num = 1;
opts.bd = 'periodic';

opts.step = @(nit) 0.5;
opts.back_track = false;
opts.tol = 1e-8; % forward solver tolerance
opts.verbose = true;
opts.check_res = true;

opts.ntit = 5;
opts.nttol = 1e-4;
opts.hjb_verbose = false;
opts.fp_verbose = false;

nxp = nx+1; dx = (rend-lend)/nx;
nyp = ny+1; dy = (rend-lend)/ny;
ntp = nt+1; dt = tend/nt;
xgrid = (lend:dx:rend);
ygrid = (lend:dy:rend);
tgrid = (0:dt:tend);
[ymesh,xmesh,tmesh] = meshgrid(ygrid,xgrid,tgrid);
xmesh0 = xmesh(:,:,1); ymesh0 = ymesh(:,:,1);

%% true
% gau = @(mu,sigma,x) 1./(sigma*sqrt(2*pi)).*exp(-(x-mu).^2./(2*sigma.^2));
% vis = 1; filename = 'mnist_vis1';
% vis = 0.3; filename = 'mnist_vis3e-1';
vis = 0.1; filename = 'mnist_vis1e-1';
% vis = 0.05; filename = 'mnist_vis5e-2';
gamma = 2; disp(['H(p)=(1/',num2str(gamma),')p^',num2str(gamma)]);
func_H = @(px,py) 1/gamma * sqrt(px.^2+py.^2).^gamma;

% f_rho0 = @(x,y) ones(size(x));
f_rho0 = @(x,y) gau(0.5,0.1,x).*gau(0.5,0.1,y);
% f_frho = @(x,y,t,rho) rho; filename = [filename,'_local'];
f_g = @(x,y,rho) zeros(size(rho));%-f_rho0(x,y);

ker_K = gau(0.5,0.1,xmesh(:,:,1)).*gau(0.5,0.2,ymesh(:,:,1)); 
f_frho = @(x,y,t,rho) 1e0*dx*dy*convn(rho.^2,ker_K,'same'); 
filename = [filename,'_nonlocal'];

% operators
op_lap = @(u) cat(1, u(end-1,  :,:) - 2*u(1,      :,:) + u(2,    :,:), ...
                     u(1:end-2,:,:) - 2*u(2:end-1,:,:) + u(3:end,:,:), ...
                     u(  end-1,:,:) - 2*u(  end,  :,:) + u(  2,  :,:) )/(dx^2)...
             +cat(2, u(:,end-1  ,:) - 2*u(:,1      ,:) + u(:,2    ,:), ...
                     u(:,1:end-2,:) - 2*u(:,2:end-1,:) + u(:,3:end,:), ...
                     u(:,  end-1,:) - 2*u(:,  end  ,:) + u(:,  2  ,:) )/(dy^2); 
op_grdx = @(u) cat(1,u(2,    :,:) - u(end-1,  :,:), ...
                     u(3:end,:,:) - u(1:end-2,:,:), ...
                     u(  2,  :,:) - u(  end-1,:,:) )/(2*dx);
op_grdy = @(u) cat(2,u(:,2    ,:) - u(:,end-1  ,:), ...
                     u(:,3:end,:) - u(:,1:end-2,:), ...
                     u(:,  2  ,:) - u(:,  end-1,:) )/(2*dy);
op_ham = @(u) func_H(op_grdx(u),op_grdy(u));

norm_sp = @(u) sqrt(sum(u.^2,'all')*dx*dy);
norm_spt = @(u) sqrt(sum(u.^2,'all')*dx*dy*dt);

%% forward problem
q_tru_norm_sp = norm_sp(q_tru);
f_f = @(x,y,t,rho) q_tru + f_frho(x,y,t,rho);
rho0 = f_rho0(xmesh0,ymesh0);

tic
[rho_tru,phi_tru,outs_tru] ...
    = ficplay2d_lf(vis,gamma,f_f,f_g,f_rho0,opts);
t_run = toc;
res_tru = outs_tru.gaps_dual(end);
disp(['forward solver residue = ', num2str(res_tru)]);
disp(num2str(outs_tru.csc_ress_rho(end)));
disp(num2str(outs_tru.sys_ress_fp(end)));

figure(3);clf;
mesh(xmesh0,ymesh0,q_tru,'LineWidth',linewidth);
for t = 1:ntp
    imshow(rho_tru(:,:,t),[]);
    colormap default
    colorbar
    pause(0.1)
end

%% measurement, potentially noisy
phi0 = phi_tru(:,:,1); 
phi0_norm_sp = norm_sp(phi0);
% dtphi0 = term0_phi - q_tru - f_frho(xgrid,0,rho0);

cst_int = dx*dy/tend/(rend-lend).^2;
term0_phi = -vis*op_lap(phi0) + op_ham(phi0) ...
            + sum(phi0,"all")*cst_int;

%% inverse general setting
% HJB parameter
opts_hjb = opts;
if isfield(opts,'hjb_verbose') opts_hjb.verbose = opts.hjb_verbose; end
% FP parameter
opts_fp = opts;
if isfield(opts,'fp_verbose') opts_fp.verbose = opts.fp_verbose; end

tol = 1e-6; % inverse tolerance
f_qinit = @(x,y) zeros(size(x));
rho_init = repmat(rho0,1,1,ntp);
q_init = f_qinit(xmesh0,ymesh0);
q_init(2:end-1,2:end-1) = rand(nx-1,ny-1);

%% inverse equilibrium correction iteration (ECI)
flag_cvg = false;

opts.Nit = 100;
if isfield(opts,'rho_num')opts = rmfield(opts,'rho_num'); end
if isfield(opts,'vx_num') opts = rmfield(opts,'vx_num'); end
if isfield(opts,'vy_num') opts = rmfield(opts,'vy_num'); end

Nit = 100;
q_nm1_eci = q_init;
% rho_nm1_eci = rho_init;

q_hist_eci = zeros(nxp,nyp,Nit);
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
    f_f = @(x,y,t,rho) q_nm1_eci + f_frho(x,y,t,rho);
    [rho_n_eci,phi_n_eci,outs_n_eci] ...
    = ficplay2d_lf(vis,gamma,f_f,f_g,f_rho0,opts);

    opts.rho_num = rho_n_eci;
    opts.vx_num = outs_n_eci.vx_num;
    opts.vy_num = outs_n_eci.vy_num;

    phi0_n_eci = phi_n_eci(:,:,1);
    term0_phi_n_eci = -vis*op_lap(phi0_n_eci) ...
                  + op_ham(phi0_n_eci) ...
                  + sum(phi0_n_eci,"all")*cst_int;

    % update q    
    res_phi_eci = term0_phi - term0_phi_n_eci;
    q_n_eci = q_nm1_eci + res_phi_eci;

    % record residue, error, number of iteration in ficplay
    nits_eci(nit+1) = nits_eci(nit) + length(outs_n_eci.gaps_dual);
    gaps_eci(nit) = outs_n_eci.gaps_dual(end);
    ress_l2_eci(nit) = norm_sp(phi0_n_eci-phi0)/phi0_norm_sp;

    err_n_eci = q_tru - q_n_eci;
    err_n_l2_eci = norm_sp(err_n_eci)/q_tru_norm_sp;
    errs_l2_eci(nit+1) = err_n_l2_eci;

    % update index
    q_nm1_eci = q_n_eci;
    % rho_nm1_eci = rho_n_eci;
    err_nm1_eci = err_n_eci;

    q_hist_eci(:,:,nit) = q_n_eci;
    
    disp(['iter ',num2str(nit), ...
        ': res ',num2str(ress_l2_eci(nit)), ...
        ', err ',num2str(errs_l2_eci(nit+1))])

    if ress_l2_eci(nit) < tol
        flag_cvg = true;
        q_hist_eci = q_hist_eci(:,:,1:nit);
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

%% plot results
close all
% marker 'o', '+', 'x', '*', '.'
% linestyle '-', '--', ':', '-.'
% color "#0072BD" (blue), 	"#D95319" (orange), "#EDB120" (yellow)
% "#7E2F8E" (purple), "#77AC30" (green), 	"#A2142F" (dark red)
img_width = 7; img_height = 6; fontsize = 12;
color_1 = "#0072BD"; marker_1 = '+'; line_1 = '-';
image_pos = [0.1, 0.1, 0.65, 0.8];   % [left, bottom, width, height]
cbar_pos = [0.78, 0.1, 0.02, 0.8];   % right of image

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
semilogy((0:nit),errs_l2_eci,'Color',color_1,...
    'LineStyle',line_1,'LineWidth',linewidth);hold on
xlabel('number of outer loop')
ylabel('$q$ rel. err.',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_relerr.eps'],...
               'BackgroundColor','none');

fig=tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
set(gcf,'unit','centimeters','Position',[7 7 img_width img_height]);% >,^,
nexttile
semilogy(ress_l2_eci,'Color',color_1,...
    'LineStyle',line_1,'LineWidth',linewidth);hold on
xlabel('number of outer loop')
ylabel('$\phi_0$ rel. err.',Interpreter='latex',FontSize=fontsize)
grid on
exportgraphics(fig,['results/',filename,'_res.eps'],...
               'BackgroundColor','none');


q_1_eci = q_hist_eci(:,:,1);

fig = imshow_fix(q_tru(2:end-1,2:end-1), ...
                 img_width,img_height);
exportgraphics(fig,['results/',filename,'_obstru.eps'],...
               'BackgroundColor','none');
close

fig = imshow_fix(phi0(2:end-1,2:end-1), ...
                 img_width,img_height);
exportgraphics(fig,['results/',filename,'_phi0.eps'],...
               'BackgroundColor','none');
close

fig = imshow_fix(term0_phi(2:end-1,2:end-1), ...
                 img_width,img_height);
exportgraphics(fig,['results/',filename,'_term_phi0.eps'],...
               'BackgroundColor','none');
close

% fig = imshow_fix(q_init(2:end-1,2:end-1), ...
%                  img_width,img_height,'duke');
% exportgraphics(fig,['results/',filename,'_obsnum0.eps'],...
%                'BackgroundColor','none');
% close

% fig = imshow_fix(q_init(2:end-1,2:end-1) - q_tru(2:end-1,2:end-1), ...
%                  img_width,img_height,'duke');
% exportgraphics(fig,['results/',filename,'_obserr0.eps'],...
%                'BackgroundColor','none');
% close

fig = imshow_fix(q_1_eci(2:end-1,2:end-1), ...
                 img_width,img_height);
exportgraphics(fig,['results/',filename,'_obsnum1.eps'],...
               'BackgroundColor','none');
close

% fig = imshow_fix(q_1_eci(2:end-1,2:end-1) - q_tru(2:end-1,2:end-1), ...
%                  img_width,img_height,'duke');
% exportgraphics(fig,['results/',filename,'_obserr1.eps'],...
%                'BackgroundColor','none');
% close

fig = imshow_fix(q_n_eci(2:end-1,2:end-1), ...
                 img_width,img_height);
exportgraphics(fig,['results/',filename,'_obsnum.eps'],...
               'BackgroundColor','none');
close

% fig = imshow_fix(q_n_eci(2:end-1,2:end-1) - q_tru(2:end-1,2:end-1), ...
%                  img_width,img_height,'duke');
% exportgraphics(fig,['results/',filename,'_obserr.eps'],...
%                'BackgroundColor','none');

