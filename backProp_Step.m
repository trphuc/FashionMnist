function [dweights,dbiases] = backProp_Step(inputs,targets,hidden_layers,neurons,weights,biases, learning_rate)
% Number of sample data
k = size(inputs,1);
[dweights,dbiases]= emptyWeights(hidden_layers,neurons);
% For each sample data 
for i = 1:k
% Feed forward
[final_hiddens,final_output] = forward_Multi(inputs(i,:)',2,weights,biases);
% Calculate error
diff= (targets(:,i) - final_output);
% Back-propagation start
% Calculate sensitive value for each layer
% s=sensitive(diff,final_output,final_hiddens{1,2},final_hiddens{1,1},weights);
s=sensitive_softmax(diff,final_output,final_hiddens{1,2},final_hiddens{1,1},weights);
% Update weight matrices
dweights{1,1} = dweights{1,1}+ learning_rate *s{1,1} * inputs(i,:);
dweights{2,1} = dweights{2,1}+ learning_rate *s{1,2} * final_hiddens{1,1}';
dweights{3,1} = dweights{3,1} + learning_rate *s{1,3} * final_hiddens{1,2}'; 
% %  Update bias
dbiases{1,1} = dbiases{1,1}+learning_rate *s{1,1};
dbiases{2,1} = dbiases{2,1} +learning_rate *s{1,2};
dbiases{3,1}= dbiases{3,1} +learning_rate *s{1,3};
% Back propagation end 
end
end
