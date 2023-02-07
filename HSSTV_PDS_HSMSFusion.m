function quality = HSSTV_PDS_HSMSFusion(image, sigma_HS, sigma_MS, lambda, mu, omega_HS, l, ratio)
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

load 'ikonos_spec_resp.mat'
MS_bands = 4;
gamma1 = 0.01;
gamma2 = 1/(200 * gamma1); 

%% observation generation %%
switch image
    case 'Reno'
        load 'Reno_band128.mat'
        mR = ikonos_sp(20:end,3:end);
        mR = cat(1, mR, zeros(size(u_org,3) - size(mR,1),4));
    case 'PaviaU'
        load 'PaviaU_band98.mat'
        mR = ikonos_sp(22:22+97,3:end);
    case 'Salinas'
        load 'Salinas_band100.mat'
        u_org = u_org(1:216,1:216,:);
        R_odd = ikonos_sp(1:2:end,:);
        R_even = ikonos_sp(2:2:end,:);
        mR = (R_odd + R_even)/2;
        mR = mR(11:end,3:end);
        mR = cat(1, mR, zeros(size(u_org,3) - size(mR,1),4));
    case 'Moffett'
        load 'MoffettField.mat'
        u_org = I_REF(1:128,1:128,:);
        R_odd = ikonos_sp(1:2:end,:);
        R_even = ikonos_sp(2:2:end,:);
        mR = (R_odd + R_even)/2;
        mR = mR(11:end,3:end);
        mR = cat(1, mR, zeros(size(u_org,3) - size(mR,1),4));
end

if gpuDeviceCount ~= 0
    GP = @(x) gpuArray(x);
    CP = @(x) gather(x);
else
    GP = @(x) x;
    CP = @(x) x;
end
    
u_org = GP(u_org - min(0,min(u_org(:))));
umax = max(abs(u_org(:)));
u_org = u_org/umax;
[rows, cols, bands] = size(u_org);
mR = mR./sum(mR,1);
mRt = mR';
sumRt = sum(mRt,1);
for i = 1:bands
    if sumRt(i) ~= 0
        mRt(:,i) = mRt(:,i)/sumRt(i);
    end
end

size_kernel=[ratio*2+1 ratio*2+1];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos = [1 1]; % The starting point of downsampling

BluKer = fspecial('gaussian',[size_kernel(1) size_kernel(2)],sig);

B = @(z) imfilter(z,BluKer,'circular');
Bt = @(z) imfilter(z,BluKer,'circular');

S = @(z) z(start_pos(1):ratio:end, start_pos(2):ratio:end,:); %downsampling
St = @(z) UpSamp(z,ratio); %upsampling
R = @(z) reshape(reshape(z,[rows*cols,bands])*mR,[rows, cols, MS_bands]);
Rt = @(z) reshape(reshape(z,[rows*cols,MS_bands])*mRt,[rows, cols, bands]);

rows2 = rows / ratio;
cols2 = cols / ratio;
v = S(B(u_org)) + sigma_HS * randn(rows2, cols2, bands);
p = R(u_org) + sigma_MS * randn(rows, cols, MS_bands);

M = Rt;
Mt = R;

temp_HS = S(B(u_org)) - v;
epsilon_HS = norm(temp_HS(:),2); % oracle radius for data-fidelity constraint
temp_MS = R(u_org) - p;
epsilon_MS = 0.98*norm(temp_MS(:),2);


%% setting %%

% difference operator with neumann boundary
Db = @(z,bands) z(:,:,[2:bands, bands])-z;
Dbt = @(z,bands) cat(3,-z(:,:,1), -z(:,:,2:bands-1) + z(:,:,1:bands-2), z(:,:,bands-1));
Dv = @(z) z([2:rows, rows],:,:) - z;
Dvt = @(z) cat(1,-z(1,:,:), -z(2:rows-1,:,:) + z(1:rows-2,:,:), z(rows-1,:,:));
Dh = @(z) z(:,[2:cols, cols],:)-z;
Dht = @(z) cat(2,-z(:,1,:), -z(:,2:cols-1,:) + z(:,1:cols-2,:), z(:,cols-1,:));

Aw = @(z,bands,omega) cat(4, Dv(Db(z,bands)), Dh(Db(z,bands)), omega * Dv(z), omega * Dh(z));
Awt = @(z,bands,omega) Dbt(Dvt(z(:,:,:,1)),bands) + Dbt(Dht(z(:,:,:,2)),bands)...
    + omega * Dvt(z(:,:,:,3)) + omega * Dht(z(:,:,:,4));
D = @(z) cat(4, Dv(z), Dh(z));
Dt = @(z) Dvt(z(:,:,:,1)) + Dht(z(:,:,:,2));

% variables
u = Bt(St(v));
q = p;
y1 = Aw(u,bands,omega_HS);
y2 = D(u) - D(M(q));
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
    u = u - gamma1 * (Awt(y1,bands,omega_HS) + Dt(y2) + Bt(St(y4)));
    q = q - gamma1 * (-Mt(Dt(y2)) + Dt(y3) + y5);
    u(u<0) = 0;
    u(u>1) = 1;
    q(q<0) = 0;
    q(q>1) = 1;

    % update of y
    y1 = y1 + gamma2 * Aw(2*u - upre,bands,omega_HS);
    switch l
        case 'L1'
            y1 = y1 - gamma2 * ProxL1norm(y1/gamma2, 1/gamma2);
        case 'L1.2'
            y1 = y1 - gamma2 * ProxTVnorm_Channelwise(y1/gamma2, 1/gamma2);
        otherwise
            error('y1 error')
    end
    y2 = y2 + gamma2 * (D(2*u - upre) - D(M(2*q - qpre)));
    y2 = y2 - gamma2 * ProxTVnorm_Channelwise(y2/gamma2, lambda/gamma2);
    y3 = y3 + gamma2 * D(2*q - qpre); 
    y3 = y3 - gamma2 * ProxTVnorm_Channelwise(y3/gamma2, mu/gamma2);
    y4 = y4 + gamma2 * S(B(2*u - upre));
    y4 = y4 - gamma2 * ProjL2ball(y4/gamma2, v, epsilon_HS);
    y5 = y5 + gamma2 * (2*q - qpre);
    y5 = y5 - gamma2 * ProjL2ball(y5/gamma2, p, epsilon_MS);
    
    % stopping condition
    res = u - upre;
    error_num = norm(res(:),2)/norm(u(:),2);
    psnr = PSNR(u, u_org);
    if rem(i,disprate) == 0
        disp(['i = ', num2str(i), ' error = ', num2str(error_num,4), ' PSNR = ', num2str(psnr,4)]);
    end
    if error_num < stopcri
        break;
    end
end

u = CP(u);
u_org = CP(u_org);
quality =  QualityIndices(u, u_org, ratio);

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
imshow(p(:,:,1:3)/max(max(max(p(:,:,1:3)))))
title('HR-MS image')
subplot(1,4,4)
imshow(u(:,:,visband)/umax_vis)
title('estimated HR-HS image')
