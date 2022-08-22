function Q = StructureMeasure(prediction,GT)
% StructureMeasure computes the similarity between the foreground map and
% ground truth(as proposed in "Structure-measure: A new way to evaluate
% foreground maps" [Deng-Ping Fan et. al - ICCV 2017])
% Usage:
%   Q = StructureMeasure(prediction,GT)
% Input:
%   prediction - Binary/Non binary foreground map with values in the range
%                [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical. 
% Output:
%   Q - The computed similarity score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input
if (~isa(prediction,'double'))
    error('The prediction should be double type...');
end
if ((max(prediction(:))>1) || min(prediction(:))<0)
    error('The prediction should be in the range of [0 1]...');
end
if (~islogical(GT))
    error('GT should be logical type...');
end

y = mean2(GT);

if (y==0)% if the GT is completely black
    x = mean2(prediction);
    Q = 1.0 - x; %only calculate the area of intersection
elseif(y==1)%if the GT is completely white
    x = mean2(prediction);
    Q = x; %only calcualte the area of intersection
else
    alpha = 0.5;
    Q = alpha*S_object(prediction,GT)+(1-alpha)*S_region(prediction,GT);
    if (Q<0)
      Q=0;
    end
end

end
