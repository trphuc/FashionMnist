function output=generateTest(dataset)
% dataset:set of sample to test
% digit: label of data_set, index of largest value of an output set 
[r,c]=size(dataset);
output = zeros(1,r);
% Load weights and bias
load('fashionTrained.mat');
load('fashion_test.mat');
% Loop through all samples
for i = 1:r
 output(2,i) = getLargestIndex(test_Multi(dataset(i,:)',2,weights, biases))-1;
 output(1,i) = ids(i,1);
end
end