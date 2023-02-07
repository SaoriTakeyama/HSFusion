function [x] = UpSamp(x,ratio)

x = permute(x,[2,1,3]);
x = upsample(x,ratio);
x = permute(x,[2,1,3]);
x = upsample(x,ratio);

% Bt = @(z) real(ifftn((fftn(z)).*bluft));
% x = Bt(x);

