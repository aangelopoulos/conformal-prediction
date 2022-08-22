function Q = S_object(prediction,GT)
% S_object Computes the object similarity between foreground maps and ground
% truth(as proposed in "Structure-measure:A new way to evaluate foreground 
% maps" [Deng-Ping Fan et. al - ICCV 2017])
% Usage:
%   Q = S_object(prediction,GT)
% Input:
%   prediction - Binary/Non binary foreground map with values in the range
%                [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical.
% Output:
%   Q - The object similarity score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the similarity of the foreground in the object level
prediction_fg = prediction;
prediction_fg(~GT)=0;
O_FG = Object(prediction_fg,GT);

% compute the similarity of the background
prediction_bg = 1.0 - prediction;
prediction_bg(GT) = 0;
O_BG = Object(prediction_bg,~GT);

% combine the foreground measure and background measure together
u = mean2(GT);
Q = u * O_FG + (1 - u) * O_BG;

end

function score = Object(prediction,GT)

% check the input 
if isempty(prediction)
    score = 0;
    return;
end
if isinteger(prediction)
    prediction = double(prediction);
end
if (~isa( prediction, 'double' ))
    error('prediction should be of type: double');
end
if ((max(prediction(:))>1) || min(prediction(:))<0)
    error('prediction should be in the range of [0 1]');
end
if(~islogical(GT))
    error('GT should be of type: logical');
end

% compute the mean of the foreground or background in prediction
x = mean2(prediction(GT));

% compute the standard deviations of the foreground or background in prediction
sigma_x = std(prediction(GT));

score = 2.0 * x./(x^2 + 1.0 + sigma_x + eps);
end