function [net_final] = testInput(inputs,hidden_weights,final_weights,hidden_bias,final_bias)
    % net output from hidden layer
    inputs=double(inputs);
    hidden_weights=double(hidden_weights);
    final_weights=double(final_weights);
    hidden_bias=double(hidden_bias);
    final_bias=double(final_bias);
    value_hidden = hidden_weights * inputs'+hidden_bias;
    % final value from hidden layer after applying transfer function
    net_hidden = sigmoid(value_hidden);
    % net output from last layer
    value_final = final_weights * net_hidden + final_bias;
    % final value from last layer after applying transfer function
    net_final = sigmoid(value_final);
end