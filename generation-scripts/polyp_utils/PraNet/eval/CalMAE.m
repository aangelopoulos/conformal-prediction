function mae = CalMAE(smap, gtImg)
% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014
if size(smap, 1) ~= size(gtImg, 1) || size(smap, 2) ~= size(gtImg, 2)
    error('Saliency map and gt Image have different sizes!\n');
end

if ~islogical(gtImg)
    gtImg = gtImg(:,:,1) > 128;
end

smap = im2double(smap(:,:,1));
fgPixels = smap(gtImg);
fgErrSum = length(fgPixels) - sum(fgPixels);
bgErrSum = sum(smap(~gtImg));
mae = (fgErrSum + bgErrSum) / numel(gtImg);