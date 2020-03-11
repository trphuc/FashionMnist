function [final_output] = test_Multi(inputs,hidden_layers,weights_array,biases_array)
% net output from hidden layer
net_hidden=inputs;
for i = 1 : hidden_layers
    value_hidden = weights_array{i,1}* net_hidden+biases_array{i,1};
    % final value from hidden layer after applying transfer function 
    tmp_hidden = sigmoid(value_hidden);
    net_hidden=tmp_hidden;
end
 % net output from last layer
value_final = weights_array{1+hidden_layers,1} * net_hidden + biases_array{1+hidden_layers,1};
% final value from last layer after applying transfer function 
final_output = sigmoid(value_final);
end