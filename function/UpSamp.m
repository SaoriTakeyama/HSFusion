function [x] = UpSamp(x,ratio)

x = permute(x,[2,1,3]);
x = upsample(x,ratio);
x = permute(x,[2,1,3]);
x = upsample(x,ratio);

end
