function Out = QualityIndices(I_HS,I_REF,umax,ratio)
%--------------------------------------------------------------------------
% Quality Indices
%
% USAGE
%   Out = QualityIndices(I_HS,I_REF,ratio)
%
% INPUT
%   I_HS    : target HS data (rows,cols,bands) normalized
%   I_REF   : reference HS data (rows,cols,bands) normalized
%   ratio   : GSD ratio between HS and MS imagers
%
% OUTPUT
%   Out.psnr : PSNR
%   Out.sam  : SAM
%   Out.ergas: ERGAS
%   Out.q2n  : Q2N
%
%--------------------------------------------------------------------------


[angle_SAM,map] = SAM(I_HS,I_REF);
Out.sam = angle_SAM;
Out.ergas = ERGAS(I_HS,I_REF,ratio);
Out.psnr = PSNR(I_REF,I_HS);
Out.sammap = map;
[Q2n_index, ~] = q2n(I_REF*umax, I_HS*umax, 32, 32);
Out.q2n = Q2n_index;

disp(['PSNR : ' num2str(Out.psnr,4)]);
disp(['SAM  : ' num2str(Out.sam,4)]);
disp(['ERGAS: ' num2str(Out.ergas,4)]);
disp(['Q2n  : ' num2str(Out.q2n,4)]);
