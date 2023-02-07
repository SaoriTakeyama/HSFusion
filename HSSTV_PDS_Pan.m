function quality = HSSTV_PDS_Pan(image, sigma_v, sigma_p, lambda, rho, omega, l, ratio)
%
% Please run "demo.m" file.
%
% Paper Information
% S. Takeyama and S. Ono, "Robust Hyperspectral Image Fusion with Simultaneous Guide Image Denoising 
%   via Constrained Convex Optimization," IEEE Trans. Geosci. Remote Sensing, vol. 60, pp. 1-18, Nov. 2022.
%

addpath Quality_Indices
addpath function

load RANDOMSTATES
rand('state', rand_state);
randn('state', randn_state);

if gpuDeviceCount ~= 0
    GP = @(x) gpuArray(x);
    CP = @(x) gather(x);
else
    GP = @(x) x;
    CP = @(x) x;
end

%% observation generation %%
switch image
    case 'Reno'
        load 'Reno_band128.mat'
        overlap = 1:73;
    case 'PaviaU'
        load 'PaviaU_band98.mat'
        overlap = 1:80;
    case 'Salinas'
        load 'Salinas_band100.mat'
        u_org = u_org(1:216,1:216,:);
        overlap = 1:36;
    case 'Moffett'
        load 'MoffettField.mat'
        u_org = I_REF(1:128,1:128,:);
        overlap = 1:41;
end

u_org = GP(u_org);
u_org = u_org - min(0,min(u_org(:)));
umax = max(abs(u_org(:)));
u_org = u_org/umax;
[rows, cols, bands] = size(u_org);
size_kernel=[ratio*2+1 ratio*2+1];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; % The starting point of downsampling
start_pos(2)=1;

BluKer = fspecial('gaussian',[size_kernel(1) size_kernel(2)],sig);
B = @(z) imfilter(z, BluKer,'circular');
Bt = @(z) imfilter(z, BluKer,'circular');

S = @(z) z(start_pos(1):ratio:end, start_pos(2):ratio:end,:);
St = @(z) Phit(z,ratio);

R = @(z) mean(z(:,:,overlap),3);

M = @(z) cat(3,repmat(z,[1,1,overlap(end)]),zeros(rows, cols, bands-overlap(end)));
Mt = @(z) sum(z(:,:,overlap),3);

Mu = @(z) cat(3, z(:,:,overlap), zeros(rows,cols,bands-size(overlap,2)));
Mut = Mu;

gamma1 = 0.005;
gamma2 = 1/(1100 * gamma1); 

v = S(B(u_org)) + sigma_v * randn(rows/ratio, cols/ratio, bands);
p = R(u_org) + sigma_p * randn(rows, cols);

temp_n = S(B(u_org)) - v;
epsilon_v = norm(temp_n(:),2); % oracle radius for data-fidelity constraint
temp_p = R(u_org) - p;
epsilon_p = norm(temp_p(:),2);


%% setting %%

% difference operator with neumann boundary
Db = @(z) z(:,:,[2:bands, bands])-z;
Dbt = @(z) cat(3,-z(:,:,1), -z(:,:,2:bands-1) + z(:,:,1:bands-2), z(:,:,bands-1));
Dv = @(z) z([2:rows, rows],:,:) - z;
Dvt = @(z) cat(1,-z(1,:,:), -z(2:rows-1,:,:) + z(1:rows-2,:,:), z(rows-1,:,:));
Dh = @(z) z(:,[2:cols, cols],:)-z;
Dht = @(z) cat(2,-z(:,1,:), -z(:,2:cols-1,:) + z(:,1:cols-2,:), z(:,cols-1,:));

Aw = @(z) cat(4, Dv(Db(z)), Dh(Db(z)), omega * Dv(z), omega * Dh(z));
Awt = @(z) Dbt(Dvt(z(:,:,:,1))) + Dbt(Dht(z(:,:,:,2))) + omega * Dvt(z(:,:,:,3)) + omega * Dht(z(:,:,:,4));
D = @(z) cat(4, Dv(z), Dh(z));
Dt = @(z) Dvt(z(:,:,:,1)) + Dht(z(:,:,:,2));

% Initialize
u = Bt(St(v));
q = p;
y1 = Aw(u);
y2 = D(Mu(u)) - D(M(q));
y3 = D(q);
y4 = S(B(u));
y5 = q;

%% main loop%%
maxIter = 10000;
stopcri = 1e-4;
disprate = 100;
for i = 1:maxIter
    
    upre = u;
    qpre = q;
    % update of u and q
    u = u - gamma1 * (Awt(y1) + Mut(Dt(y2)) + Bt(St(y4)));
    q = q - gamma1 * (-Mt(Dt(y2)) + Dt(y3) + y5);
    u(u<0) = 0;
    u(u>1) = 1;
    q(q<0) = 0;
    q(q>1) = 1;

    % update of y
    y1 = y1 + gamma2 * Aw(2*u - upre);
    switch l
        case 'L1'
            y1 = y1 - gamma2 * ProxL1norm(y1/gamma2, 1/gamma2);
        case 'L1.2'
            y1 = y1 - gamma2 * ProxTVnorm_Channelwise(y1/gamma2, 1/gamma2);
    end
    y2 = y2 + gamma2 * (D(Mu(2*u - upre)) - D(M(2*q - qpre)));
    y2 = y2 - gamma2 * ProxTVnorm_Channelwise(y2/gamma2, lambda/gamma2);
    y3 = y3 + gamma2 * D(2*q - qpre);
    y3 = y3 - gamma2 * ProxTVnorm_Channelwise(y3/gamma2, rho/gamma2);
    y4 = y4 + gamma2 * S(B(2*u - upre));
    y4 = y4 - gamma2 * ProjL2ball(y4/gamma2, v, epsilon_v);
    y5 = y5 + gamma2 * (2*q - qpre);
    y5 = y5 - gamma2 * ProjL2ball(y5/gamma2, p, epsilon_p);
    
    % stopping condition
    res = u - upre;
    error = norm(res(:),2)/norm(u(:),2);
    if rem(i,disprate) == 0
        psnr = PSNR(u, u_org);
        disp(['i = ', num2str(i), ' error = ', num2str(error,4), ' PSNR = ', num2str(psnr,4)]);
    end
    if error < stopcri
        break;
    end
end
%% result plot
u = CP(u);
u_org = CP(u_org);
umax = CP(umax);
quality = QualityIndices(u, u_org, umax, ratio);

visband = [8 16 32];
umax_vis = max(max(max(u_org(:,:,visband))));
figure
subplot(1,4,1)
imshow(u_org(:,:,visband)/umax_vis)
title('Ground-truth')
subplot(1,4,2)
imshow(v(:,:,visband)/umax_vis)
title('LR-HS image')
subplot(1,4,3)
imshow(p/max(p(:)))
title('PAN image')
subplot(1,4,4)
imshow(u(:,:,visband)/umax_vis)
title('estimated HR-HS image')
end
