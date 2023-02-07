clear
close all

% Settings
image = 'PaviaU'; % select 'Reno', 'PaviaU', 'Salinas', or 'Moffett'
L = 'L1'; % select 'L1' or 'L1.2'
Guide_type = 'PAN'; %select 'PAN' or 'MS'
omega = 0.01; % hyperparameter in HSSTV
lambda = 0.03; 
rho = 1.0;
sigma_HS = 0.1; % noise intensity
sigma_guide = 0.04;
r = 8; % downsampling ratio

% Fusion
switch Guide_type
    case 'PAN'
        QI = HSSTV_PDS_Pan(image, sigma_HS, sigma_guide, lambda, rho, omega, L, r);
    case 'MS'
        QI = HSSTV_PDS_HSMSFusion(image, sigma_HS, sigma_guide, lambda, rho, omega, L, r);
end