function [weights,biases] = emptyWeights(hidden_layers,neurons)
weights{hidden_layers+1,1}=[];
biases{hidden_layers+1,1}=[];
for i = 1:hidden_layers+1
weight = zeros(neurons(i+1),neurons(i));
weights{i,1}=weight;
bias = zeros(neurons(i+1),1);
biases{i,1}=bias;
end
end

