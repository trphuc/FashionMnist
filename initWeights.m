function [weights,biases] = initWeights(hidden_layers,neurons)
weights{hidden_layers+1,1}=[];
biases{hidden_layers+1,1}=[];
for i = 1:hidden_layers+1
weight = rand(neurons(i+1),neurons(i))-0.5;
weights{i,1}=weight;
bias = rand(neurons(i+1),1)-0.5;
biases{i,1}=bias;
end
end

