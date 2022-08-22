%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation tool boxs for PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI20).
%Author: Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao
%Homepage: http://dpfan.net/
%Projectpage: https://github.com/DengPingFan/PraNet
%First version: 2020-6-28
%Any questions please contact with dengpfan@gmail.com.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: Providing several important metrics: Dice, IoU, F1, S-m (ICCV'17), Weighted-F1 (CVPR'14)
%          E-m (IJCAI'18), Precision, Recall, Sensitivity, Specificity, MAE.


clear all;
close all;
clc;

% ---- 1. ResultMap Path Setting ----
ResultMapPath = '../results/';
Models = {'PraNet'}; %{'UNet','UNet++','PraNet','SFA'};
modelNum = length(Models);

% ---- 2. Ground-truth Datasets Setting ----
DataPath = '../data/TestDataset/';
Datasets = {'CVC-300','CVC-ClinicDB'}; %{'CVC-ClinicDB', 'CVC-ColonDB','ETIS-LaribPolypDB', 'Kvasir','CVC-300'};

% ---- 3. Evaluation Results Save Path Setting ----
ResDir = './EvaluateResults/';
ResName='_result.txt';  % You can change the result name.

Thresholds = 1:-1/255:0;
datasetNum = length(Datasets);

for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d}   % print cur dataset name
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    ResPath = [ResDir dataset '-mat/']; % The result will be saved in *.mat file so that you can used it for the next time.
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset ResName];  % The evaluation result will be saved in `../Resluts/Result-XXXX` folder.
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m}   % print cur model name
        
        gtPath = [DataPath dataset '/masks/'];
        resMapPath = [ResultMapPath '/' model '/' dataset '/'];
        
        imgFiles = dir([resMapPath '*.png']);
        imgNUM = length(imgFiles);
        
        [threshold_Fmeasure, threshold_Emeasure, threshold_IoU] = deal(zeros(imgNUM,length(Thresholds)));
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        [threshold_Sensitivity, threshold_Specificity, threshold_Dice] = deal(zeros(imgNUM,length(Thresholds)));
        
        [Smeasure, wFmeasure, MAE] =deal(zeros(1,imgNUM));
        
        for i = 1:imgNUM
            name =  imgFiles(i).name;
            fprintf('Evaluating(%s Dataset,%s Model, %s Image): %d/%d\n',dataset, model, name, i,imgNUM);
            
            %load gt
            gt = imread([gtPath name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load resMap
            resmap  = imread([resMapPath name]);
            %check size
            if size(resmap, 1) ~= size(gt, 1) || size(resmap, 2) ~= size(gt, 2)
                resmap = imresize(resmap,size(gt));
                imwrite(resmap,[resMapPath name]);
                fprintf('Resizing have been operated!! The resmap size is not math with gt in the path: %s!!!\n', [resMapPath name]);
            end
            
            resmap = im2double(resmap(:,:,1));
            
            %normalize resmap to [0, 1]
            resmap = reshape(mapminmax(resmap(:)',0,1),size(resmap));
            
            % S-meaure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
            Smeasure(i) = StructureMeasure(resmap,logical(gt));
            
            % Weighted F-measure metric published in CVPR'14 (How to evaluate the foreground maps?)
            wFmeasure(i) = original_WFb(resmap,logical(gt));
            
            MAE(i) = mean2(abs(double(logical(gt)) - resmap));
            
            [threshold_E, threshold_F, threshold_Pr, threshold_Rec, threshold_Iou]  = deal(zeros(1,length(Thresholds)));
            [threshold_Spe, threshold_Dic]  = deal(zeros(1,length(Thresholds)));
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), threshold_Dic(t), threshold_F(t), threshold_Iou(t)] = Fmeasure_calu(resmap,double(gt),size(gt),threshold);
                
                Bi_resmap = zeros(size(resmap));
                Bi_resmap(resmap>threshold)=1;
                threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
            end
            
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Sensitivity(i,:) = threshold_Rec;
            threshold_Specificity(i,:) = threshold_Spe;
            threshold_Dice(i,:) = threshold_Dic;
            threshold_IoU(i,:) = threshold_Iou;
            
        end
        
        %MAE
        mae = mean2(MAE);
        
        %Sm
        Sm = mean2(Smeasure);
        
        %wFm
        wFm = mean2(wFmeasure);
        
        %E-m
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        %Sensitivity
        column_Sen = mean(threshold_Sensitivity,1);
        meanSen = mean(column_Sen);
        maxSen = max(column_Sen);
        
        %,Specificity
        column_Spe = mean(threshold_Specificity,1);
        meanSpe = mean(column_Spe);
        maxSpe = max(column_Spe);
        
        %Dice
        column_Dic = mean(threshold_Dice,1);
        meanDic = mean(column_Dic);
        maxDic = max(column_Dic);
        
        %IoU
        column_IoU = mean(threshold_IoU,1);
        meanIoU = mean(column_IoU);
        maxIoU = max(column_IoU);
        
        save([ResPath model],'Sm', 'mae', 'column_Dic', 'column_Sen', 'column_Spe', 'column_E','column_IoU','maxDic','maxEm','maxSen','maxSpe','maxIoU','meanIoU','meanDic','meanEm','meanSen','meanSpe');
        fprintf(fileID, '(Dataset:%s; Model:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',dataset,model,meanDic,meanIoU,wFm,Sm,meanEm,mae,maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
        fprintf('(Dataset:%s; Model:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',dataset,model,meanDic,meanIoU,wFm,Sm,meanEm,mae,maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
    end
    
    toc;
end




