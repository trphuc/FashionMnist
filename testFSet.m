function [percentage]=testFSet(inputs, targets)
% dataset:set of sample to test
% digit: label of data_set, index of largest value of an output set 
success=0;
[r,c]=size(inputs);
% Load weights and bias
load('fashionTrained.mat');
% Loop through all samples
for i = 1:r
    t=test_Multi(inputs(i,:)',2,weights, biases);
    if(getLargestIndex(t)==getLargestIndex(targets(:,i)))
        success=success+1; 
    end
end
percentage=(success/r)*100;
end