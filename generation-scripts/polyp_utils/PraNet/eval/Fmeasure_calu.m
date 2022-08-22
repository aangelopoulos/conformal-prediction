%%
function [PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU] = Fmeasure_calu(sMap, gtMap, gtsize, threshold)
%threshold =  2* mean(sMap(:)) ;
if ( threshold > 1 )
    threshold = 1;
end

Label3 = zeros( gtsize );
Label3( sMap>=threshold ) = 1;

NumRec = length( find( Label3==1 ) ); %FP+TP
NumNoRec = length(find(Label3==0)); % FN+TN
LabelAnd = Label3 & gtMap;
NumAnd = length( find ( LabelAnd==1 ) ); %TP
num_obj = sum(sum(gtMap));  %TP+FN
num_pred = sum(sum(Label3)); % FP+TP

FN = num_obj-NumAnd;
FP = NumRec-NumAnd;
TN = NumNoRec-FN;

%SpecifTem = TN/(TN+FP)
%Precision = TP/(TP+FP)

if NumAnd == 0
    PreFtem = 0;
    RecallFtem = 0;
    FmeasureF = 0;
    Dice = 0;
    SpecifTem = 0;
    IoU = 0;
else
    IoU = NumAnd/(FN+NumRec);  %TP/(FN+TP+FP)
    PreFtem = NumAnd/NumRec;
    RecallFtem = NumAnd/num_obj;
    SpecifTem = TN/(TN+FP);
    Dice = 2 * NumAnd/(num_obj+num_pred);
%     FmeasureF = ( ( 1.3* PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) ); % beta = 0.3
    FmeasureF = (( 2.0 * PreFtem * RecallFtem ) / (PreFtem + RecallFtem)); % beta = 1.0
end

%Fmeasure = [PreFtem, RecallFtem, FmeasureF];

