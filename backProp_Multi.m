function [weights,biases] = backProp_Multi(inputs,targets,layers,input_neurons,hidden_neurons,hidden_neurons2,output_neurons,epoch, learning_rate)
% Number of sample data
k = size(inputs,1);
arrayCost = zeros(epoch,1);
arrayEpoch = zeros(epoch,1);
%Initialize weight and bias matrices
hidden_weights = rand(hidden_neurons,input_neurons)-0.5;
hidden_weights2 = rand(hidden_neurons2,hidden_neurons)-0.5; 
final_weights = rand(output_neurons,hidden_neurons2)-0.5;
hidden_bias = rand(hidden_neurons,1)-0.5;
hidden_bias2 = rand(hidden_neurons2,1)-0.5; 
outputbias = rand(output_neurons,1)-0.5;

% Hold average cost
avgCost = 0;
% Iteration epoch number 
for j = 1:epoch
j
% For each sample data 
for i = 1:k
% Feed forward
weights={hidden_weights;hidden_weights2;final_weights};
biases={hidden_bias;hidden_bias2;outputbias};
[final_hiddens,final_output] = forward_Multi(inputs(i,:)',2,weights,biases);
final_hidden = final_hiddens{1,1};
final_hidden2 = final_hiddens{1,2};

% Calculate error
diff= (targets(:,i) - final_output);
cost=0;
for u = 1:length(diff)
cost = cost + diff(u) ^ 2; 
end
avgCost = avgCost + cost;
% Back-propagation start
% Calculate sensitive value for each layer
s_output = -2*diff .* (final_output .* (1-final_output)); 
s_hidden2 = (final_hidden2 .* (1-final_hidden2)).*(final_weights'*s_output );
s_hidden = (final_hidden .* (1-final_hidden)).*(hidden_weights2'*s_hidden2 );
% Update weight matrices
hidden_weights = hidden_weights - learning_rate .* s_hidden * inputs(i,:);
hidden_weights2 = hidden_weights2 - learning_rate .* s_hidden2 * final_hidden';
final_weights = final_weights - learning_rate .*s_output * final_hidden2'; 
% Update bias
hidden_bias = hidden_bias -learning_rate*s_hidden;
hidden_bias2 = hidden_bias2 -learning_rate*s_hidden2;
outputbias = outputbias -learning_rate*s_output;
% Back propagation end 
end
% Calculate mean square error
avgCost =avgCost/k;
arrayCost(j)=avgCost;
avgCost = 0;
arrayEpoch(j)=j; 
weights={hidden_weights;hidden_weights2;final_weights};
biases={hidden_bias;hidden_bias2;outputbias};
end
