function [weights,biases] = backProp_Multi(inputs,targets,input_neurons,hidden_neurons,hidden_neurons2,output_neurons,epoch, learning_rate)
% Number of sample data
k = size(inputs,1);
arrayCost = zeros(epoch,1);
neurons=[input_neurons,hidden_neurons,hidden_neurons2,output_neurons];
[weights,biases]=initWeights(2,neurons);
% Hold average cost
% Iteration epoch number 
for j = 1:epoch
j
avgCost = 0;
% For each sample data 
for i = 1:k
% Feed forward
[final_hiddens,final_output] = forward_Multi(inputs(i,:)',2,weights,biases);
% Calculate error
diff= (targets(:,i) - final_output);
% cost=0;
% for u = 1:length(diff)
% cost = cost + diff(u);
% end
% avgCost = avgCost + cost*cost/2;
% Cross entropy error
cost=crossentropy_loss(targets(:,i),final_output);
avgCost = avgCost + cost;
% Back-propagation start
% Calculate sensitive value for each layer
s=sensitive_softmax(diff,final_output,final_hiddens{1,2},final_hiddens{1,1},weights);
% Update weight matrices
weights{1,1} = weights{1,1} - learning_rate .* s{1,1} * inputs(i,:);
weights{2,1} = weights{2,1} - learning_rate .* s{1,2} * final_hiddens{1,1}';
weights{3,1} = weights{3,1} - learning_rate .*s{1,3} * final_hiddens{1,2}'; 
% Update bias
biases{1,1} = biases{1,1} -learning_rate*s{1,1};
biases{2,1} = biases{2,1} -learning_rate*s{1,2};
biases{3,1}= biases{3,1} -learning_rate*s{1,3};
% Back propagation end 
end
% Calculate mean square error
avgCost =avgCost/k
arrayCost(j)=avgCost;
avgCost = 0;
end
end