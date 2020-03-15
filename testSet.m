function [percentage]=testSet(data_set, digit)
% dataset:set of sample to test
% digit: label of data_set, index of largest value of an output set 
success=0;
[r,c]=size(data_set);
% Load weights and bias
load('mnistTrained.mat');
load('mnist_all.mat');
% Loop through all samples
for i = 1:r
 t =testInput(data_set(i,:),hidden_weights,final_weights,hidden_bias,final_bias); if(getLargestIndex(t)==digit)
success=success+1; end
end
percentage=(success/r)*100;
end

