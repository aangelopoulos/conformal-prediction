function Q = S_region(prediction,GT)
% S_region computes the region similarity between the foreground map and
% ground truth(as proposed in "Structure-measure:A new way to evaluate
% foreground maps" [Deng-Ping Fan et. al - ICCV 2017])
% Usage:
%   Q = S_region(prediction,GT)
% Input:
%   prediction - Binary/Non binary foreground map with values in the range
%                [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical.
% Output:
%   Q - The region similarity score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find the centroid of the GT
[X,Y] = centroid(GT);

% divide GT into 4 regions
[GT_1,GT_2,GT_3,GT_4,w1,w2,w3,w4] = divideGT(GT,X,Y);

%Divede prediction into 4 regions
[prediction_1,prediction_2,prediction_3,prediction_4] = Divideprediction(prediction,X,Y);

%Compute the ssim score for each regions
Q1 = ssim(prediction_1,GT_1);
Q2 = ssim(prediction_2,GT_2);
Q3 = ssim(prediction_3,GT_3);
Q4 = ssim(prediction_4,GT_4);

%Sum the 4 scores
Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4;

end

function [X,Y] = centroid(GT)
% Centroid Compute the centroid of the GT
% Usage:
%   [X,Y] = Centroid(GT)
% Input:
%   GT - Binary ground truth. Type: logical.
% Output:
%   [X,Y] - The coordinates of centroid.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[rows,cols] = size(GT);

if(sum(GT(:))==0)
    X = round(cols/2);
    Y = round(rows/2);
else     
    total=sum(GT(:));
    i=1:cols;
    j=(1:rows)';
    X=round(sum(sum(GT,1).*i)/total);
    Y=round(sum(sum(GT,2).*j)/total);
    
    %dGT = double(GT); 
    %x = ones(rows,1)*(1:cols);
    %y = (1:rows)'*ones(1,cols);
    %area = sum(dGT(:));
    %X = round(sum(sum(dGT.*x))/area);
    %Y = round(sum(sum(dGT.*y))/area);
end

end

% divide the GT into 4 regions according to the centroid of the GT and return the weights
function [LT,RT,LB,RB,w1,w2,w3,w4] = divideGT(GT,X,Y)
% LT - left top;
% RT - right top;
% LB - left bottom;
% RB - right bottom;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%width and height of the GT
[hei,wid] = size(GT);
area = wid * hei;

%copy the 4 regions 
LT = GT(1:Y,1:X);
RT = GT(1:Y,X+1:wid);
LB = GT(Y+1:hei,1:X);
RB = GT(Y+1:hei,X+1:wid);

%The different weight (each block proportional to the GT foreground region).
w1 = (X*Y)./area;
w2 = ((wid-X)*Y)./area;
w3 = (X*(hei-Y))./area;
w4 = 1.0 - w1 - w2 - w3;
end

%Divide the prediction into 4 regions according to the centroid of the GT 
function [LT,RT,LB,RB] = Divideprediction(prediction,X,Y)

%width and height of the prediction
[hei,wid] = size(prediction);

%copy the 4 regions 
LT = prediction(1:Y,1:X);
RT = prediction(1:Y,X+1:wid);
LB = prediction(Y+1:hei,1:X);
RB = prediction(Y+1:hei,X+1:wid);

end

function Q = ssim(prediction,GT)
% ssim computes the region similarity between foreground maps and ground
% truth(as proposed in "Structure-measure: A new way to evaluate foreground 
% maps" [Deng-Ping Fan et. al - ICCV 2017])
% Usage:
%   Q = ssim(prediction,GT)
% Input:
%   prediction - Binary/Non binary foreground map with values in the range
%                [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical.
% Output:
%   Q - The region similarity score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dGT = double(GT);

[hei,wid] = size(prediction);
N = wid*hei;

%Compute the mean of SM,GT
x = mean2(prediction);
y = mean2(dGT);

%Compute the variance of SM,GT
sigma_x2 = sum(sum((prediction - x).^2))./(N - 1 + eps);%sigma_x2 = var(prediction(:))
sigma_y2 = sum(sum((dGT - y).^2))./(N - 1 + eps);       %sigma_y2 = var(dGT(:));      

%Compute the covariance between SM and GT
sigma_xy = sum(sum((prediction - x).*(dGT - y)))./(N - 1 + eps);

alpha = 4 * x * y * sigma_xy;
beta = (x.^2 + y.^2).*(sigma_x2 + sigma_y2);

if(alpha ~= 0)
    Q = alpha./(beta + eps);
elseif(alpha == 0 && beta == 0)
    Q = 1.0;
else
    Q = 0;
end

end
