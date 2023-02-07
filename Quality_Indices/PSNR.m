function [out] = PSNR(u, u_org)

MSE = sum(sum(sum((u - u_org).^2)));
MSE= MSE/(numel(u));
out = 10 * log10(1/MSE);

end