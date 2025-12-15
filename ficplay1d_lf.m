function [rho_k,phi_hat_kp1,outs] ...
    = ficplay1d_lf(vis,gamma,func_f,func_g,func_rho_init,opts)
% fictitious play for mean-field game problem
% vis, viscosity
% gamma, H(x,p) = 1/gamma*|p|^gamma
% func_f, interaction cost
% func_g, terminal cost
% func_rho0, initial density
% opts: 
flag_cvg = false;
flag_divg = false;
%% ficplay parameter
if isfield(opts,'Nit') Nit = opts.Nit; else Nit = 100; end
if isfield(opts,'tol') tol = opts.tol; else tol = 1e-6; end
if isfield(opts,'verbose') verbose = opts.verbose; else verbose = false; end
if isfield(opts,'check_res') check_res = opts.check_res; else check_res = false; end
if isfield(opts,'func_rho_true') || isfield(opts,'rho_eq')
    have_exact = true;
else
    have_exact = false;
end
% parameters for line search
if isfield(opts,'back_track') && opts.back_track
    step = @(nit) 1;
    if isfield(opts,'subNit') subNit = opts.subNit; else subNit = 5; end
    if isfield(opts,'alpha') alpha = opts.alpha; else alpha = 0.5; end
    if isfield(opts,'zeta') zeta = opts.zeta; else zeta = 0.8; end
else
    if isfield(opts,'step') 
        step = opts.step; 
    else 
        step = @(nit) 2/(nit+2); 
    end
    subNit = 1; alpha = 1; zeta = 1;
end
%% HJB parameter
opts_hjb = opts;
if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    opts_hjb.bd = opts.bd; 
    % opts_hjb.phil = opts.func_phil(tgrid);
    % opts_hjb.phir = opts.func_phir(tgrid);
elseif isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    opts_hjb.bd = 'periodic';
else % default boundary condition
    opts.bd = 'neumann';
    opts_hjb.bd = 'neumann';
end
if isfield(opts,'hjb_verbose') opts_hjb.verbose = opts.hjb_verbose; end

%% FP parameter
opts_fp = opts;
if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    opts_fp.bd = opts.bd; 
    % opts_fp.rhol = opts.func_rhol(tgrid);
    % opts_fp.rhor = opts.func_rhor(tgrid);
elseif isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    opts_fp.bd = opts.bd;
else % default boundary condition
    opts_fp.bd = 'zero_flux';
end
if isfield(opts,'fp_verbose') opts_fp.verbose = opts.fp_verbose; end

%% system parameter
gammap = gamma/(gamma-1);
func_L = @(v) 1/gammap*abs(v).^gammap;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);
if isfield(opts,'xrange') lend = opts.xrange(1); rend = opts.xrange(2);
                     else lend = 0; rend = 1; end
if isfield(opts,'tend') tend = opts.tend; else tend = 1; end
if isfield(opts,'phi_num')
    phi_km1 = opts.phi_num;
    nxp = size(phi_km1,1); nx = nxp-1; dx = (rend-lend)/nx;
    ntp = size(phi_km1,2); nt = ntp-1; dt = tend/nt;
    xgrid = (lend:dx:rend)';
    tgrid = (0:dt:tend);
    [tmesh,xmesh] = meshgrid(tgrid,xgrid);
    rho0 = func_rho_init(xgrid);
    if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
        opts_fp.rhol = opts.func_rhol(tgrid);
        opts_fp.rhor = opts.func_rhor(tgrid);
    end
    if strcmp(opts.bd,'periodic')
        Dphi = cat(1,phi_km1(2,:)-phi_km1(end,:),...
                     phi_km1(3:end,:)-phi_km1(1:end-2,:),...
                     phi_km1(1,:)-phi_km1(end-1,:))/(2*dx);
    else
        Dphi = cat(1,phi_km1(2,:)-phi_km1(1,:),...
                     phi_km1(3:end,:)-phi_km1(1:end-2,:),...
                     phi_km1(end,:)-phi_km1(end-1,:))/(2*dx);
    end
    v_km1 = - grad_H(Dphi);
    rho_km1 = fp1d_lf(rho0,v_km1,vis,opts_fp);
    rho_km1 = max(rho_km1,0);
elseif isfield(opts,'rho_num')
    rho_km1 = opts.rho_num;
    nxp = size(rho_km1,1); nx = nxp-1; dx = (rend-lend)/nx;
    ntp = size(rho_km1,2); nt = ntp-1; dt = tend/nt;
    if isfield(opts,'v_num')
        v_km1 = opts.v_num;
    else
        v_km1 = zeros(nxp,ntp);
    end
    xgrid = (lend:dx:rend)';
    tgrid = (0:dt:tend);
    [tmesh,xmesh] = meshgrid(tgrid,xgrid);
    rho0 = rho_km1(:,1);
else
    if isfield(opts,'nt') nt = opts.nt; else nt = 2^5; end
    if isfield(opts,'nx') nx = opts.nx; 
    else nx = 2^5*2^ceil(log((rend-lend)/tend)/log(2)); end
    nxp = nx+1; dx = (rend-lend)/nx;
    ntp = nt+1; dt = tend/nt;
    xgrid = (lend:dx:rend)';
    tgrid = (0:dt:tend);
    [tmesh,xmesh] = meshgrid(tgrid,xgrid);
    rho0 = func_rho_init(xgrid);
    rho_km1 = repmat(rho0,1,ntp);
    v_km1 = zeros(nxp,ntp);
end
if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    opts_fp.rhol = opts.func_rhol(tgrid);
    opts_fp.rhor = opts.func_rhor(tgrid);
    opts_hjb.phil = opts.func_phil(tgrid);
    opts_hjb.phir = opts.func_phir(tgrid);
end

inprod_QT = @(x,y) sum(x.*y,'all')*dx*dt;
inprod_Omega = @(x,y) sum(x.*y,'all')*dx;
l2norm_QT = @(x) sqrt(sum(x.^2,'all')*dx*dt);

%% main iteration (prepare)
gaps_dual = zeros(Nit,1);
csc_ress_rho = zeros(Nit,1);
steps = zeros(Nit,1);
if check_res
    sys_ress_hjb = zeros(Nit,1);
    sys_ress_fp = zeros(Nit,1);
end
if have_exact
    norms_diff_rho = zeros(Nit,1);
    norms_diff_rhohat = zeros(Nit,1);
    angles = zeros(Nit,1);        
    if isfield(opts,'func_rho_true')
        rho_eq = opts.func_rho_true(xmesh,tmesh);
    elseif isfield(opts,'rho_eq')
        rho_eq = opts.rho_eq;
    end
end

%% main iteration
%--------- HJB
f_km1 = func_f(xmesh,tmesh,rho_km1);
g_km1 = func_g(xgrid,rho_km1(:,end));
phi_hat_k = hjb1d_revt_lf(g_km1,vis,gamma,f_km1,opts_hjb);

%--------- FP
if strcmp(opts.bd,'periodic')
    Dphi = cat(1,phi_hat_k(2,:)-phi_hat_k(end,:),...
                 phi_hat_k(3:end,:)-phi_hat_k(1:end-2,:),...
                 phi_hat_k(1,:)-phi_hat_k(end-1,:))/(2*dx);
else
    Dphi = cat(1,phi_hat_k(2,:)-phi_hat_k(1,:),...
                 phi_hat_k(3:end,:)-phi_hat_k(1:end-2,:),...
                 phi_hat_k(end,:)-phi_hat_k(end-1,:))/(2*dx);
end
v_hat_k = - grad_H(Dphi);
rho_hat_k = fp1d_lf(rho0,v_hat_k,vis,opts_fp);
gaps_dual_km1 = comp_dualgap(rho_km1,v_km1,rho_hat_k,v_hat_k,f_km1,g_km1);

for nit = 1:Nit
    % fic play 
    dlt = step(nit);
    for subnit = 1:subNit
        %--------- density average
        dlt = dlt * alpha;
        rho_k = (1-dlt)*rho_km1 + dlt*rho_hat_k;

        %--------- HJB
        f_k = func_f(xmesh,tmesh,rho_k);
        g_k = func_g(xgrid,rho_k(:,end));
        phi_hat_kp1 = hjb1d_revt_lf(g_k,vis,gamma,f_k,opts_hjb);
    
        %--------- FP
        if strcmp(opts.bd,'periodic')
            Dphi = cat(1,phi_hat_kp1(2,:)-phi_hat_kp1(end,:),...
                         phi_hat_kp1(3:end,:)-phi_hat_kp1(1:end-2,:),...
                         phi_hat_kp1(1,:)-phi_hat_kp1(end-1,:))/(2*dx);
        else
            Dphi = cat(1,phi_hat_kp1(2,:)-phi_hat_kp1(1,:),...
                         phi_hat_kp1(3:end,:)-phi_hat_kp1(1:end-2,:),...
                         phi_hat_kp1(end,:)-phi_hat_kp1(end-1,:))/(2*dx);
        end
        v_hat_kp1 = - grad_H(Dphi);
        rho_hat_kp1 = fp1d_lf(rho0,v_hat_kp1,vis,opts_fp);

        diff_f = f_km1 - f_k;
        diff_g = g_km1 - g_k; 
        diff_rhohat = rho_hat_k - rho_hat_kp1;
        D = -inprod_QT(diff_f,diff_rhohat) ...
            -inprod_Omega(diff_g,diff_rhohat(:,end));
        if D <= zeta*dlt*gaps_dual_km1
            break
        end

    end
    steps(nit) = dlt;
    v_k = (1-dlt)*v_km1 + dlt*v_hat_k;

    %--------- residuals
    %--------- compute dual gap (exploitability)
    gaps_dual_k = comp_dualgap(rho_k,v_k,rho_hat_kp1,v_hat_kp1,f_k,g_k);
    gaps_dual(nit) = gaps_dual_k;
    %--------- consecutive residual
    csc_res_rho = rho_hat_kp1 - rho_k; 
    csc_ress_rho(nit) = l2norm_QT(csc_res_rho); 
    if verbose 
        disp('nit, dual_gap, csc_res_rho');
        disp([nit,gaps_dual_k,csc_ress_rho(nit)]); 
    end
    %--------- system residual
    if check_res
        [sys_res_hjb,sys_res_fp,sys_res_outs] ...
            = comp_res1d_lf(xmesh,tmesh,rho_k,phi_hat_kp1,...
                            vis,opts.vis_num,gamma,func_f,func_g);
        sys_ress_hjb(nit) = sys_res_hjb;
        sys_ress_fp(nit) = sys_res_fp;
    end
    %--------- error
    if have_exact
        diff_rho = rho_k - rho_eq;
        diff_rho_hat = rho_hat_kp1 - rho_eq;
        norm_diff_rho = l2norm_QT(diff_rho);
        norm_diff_rhohat = l2norm_QT(diff_rho_hat);
        inprod = inprod_QT(diff_rho,diff_rho_hat);
        norms_diff_rho(nit) = norm_diff_rho;
        norms_diff_rhohat(nit) = norm_diff_rhohat;
        angles(nit) = inprod/(norm_diff_rho*norm_diff_rhohat);
    end
    
    if isnan(csc_ress_rho(nit))
        disp([num2str(nit),'-th iteration unstable'])
        break
    end
    if abs(gaps_dual(nit)) < tol
        flag_cvg = true;
        disp(['fictitious play converges at ',num2str(nit),'-th iteration'])
        break
    elseif (csc_ress_rho(nit) > 1e6) || (gaps_dual(nit) > 1e6) 
        flag_divg = true;
        disp(['fictitious play diverges at ',num2str(nit),'-th iteration'])
        break
    end

    %--------- update
    rho_km1 = rho_k;
    rho_hat_k = rho_hat_kp1;
    v_km1 = v_k;
    v_hat_k = v_hat_kp1;
    gaps_dual_km1 = gaps_dual_k;
    f_km1 = f_k;
    g_km1 = g_k;
end
%% copy results
if ~flag_cvg && ~flag_divg
    disp(['fictitious play finishes at ',num2str(nit),'-th iteration'])
end
outs.v_num = v_k;
outs.gaps_dual = gaps_dual(1:nit);
outs.steps = steps(1:nit);
outs.csc_ress_rho = csc_ress_rho(1:nit);

outs.xmesh = xmesh;
outs.tmesh = tmesh;
if check_res
    outs.sys_ress_hjb = sys_ress_hjb(1:nit);
    outs.sys_ress_fp = sys_ress_fp(1:nit);
    outs.sys_res_hjb_mesh = sys_res_outs.res_hjb_mesh;
    outs.sys_res_fp_mesh = sys_res_outs.res_fp_mesh;
end
if have_exact
    outs.norms_diff_rho = norms_diff_rho(1:nit);
    outs.norms_diff_rhohat = norms_diff_rhohat(1:nit);
    outs.angles = angles(1:nit);
end

%% 
    function dyncost = comp_dyncost(rho,v)
        dyncost = inprod_QT(rho(:,2:end), func_L(v(:,1:end-1)));
    end

    function gap_dual = comp_dualgap(rho,v,rho_hat,v_hat,f,g)
        gap_dual = comp_dyncost(rho,v) - comp_dyncost(rho_hat,v_hat);
        % gap_dual = gap_dual + dt*dx*sum(f.*(rho-rho_hat),'all');
        % gap_dual = gap_dual + dx*sum(g.*(rho(:,end)-rho_hat(:,end)),'all');
        gap_dual = gap_dual + inprod_QT(f, rho-rho_hat);
        gap_dual = gap_dual + inprod_Omega(g, rho(:,end)-rho_hat(:,end));
    end

end