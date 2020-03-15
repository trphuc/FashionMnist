function [hidden_weights,final_weights,hidden_bias,output_bias] =  backProp(inputs,targets,input_neurons,hidden_neurons,output_neurons,epoch, learning_rate)
    % Number of sample data
    k = size(inputs,1);
    arrayCost = zeros(epoch,1);
    arrayEpoch = zeros(epoch,1);
    %Initialize weight and bias matrices
    hidden_weights = rand(hidden_neurons,input_neurons)-0.5;
    final_weights =  rand(output_neurons,hidden_neurons)-0.5;
    hidden_bias = rand(hidden_neurons,1)-0.5;
    output_bias = rand(output_neurons,1)-0.5;
    % Hold average cost
    avgCost = 0;
    size(arrayCost)
    % Iteration epoch number
    for j = 1:epoch
        % For each sample data
        for i = 1:k
            % Feed forward
            [final_hidden,final_output] = forward(inputs(i,:),hidden_weights,final_weights,hidden_bias,output_bias);
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
            s_hidden =  (final_hidden .* (1-final_hidden)).* (final_weights'*s_output );
            % Update weight matrices
            hidden_weights = hidden_weights - learning_rate .* s_hidden * inputs(i,:);
            final_weights = final_weights - learning_rate .*s_output * final_hidden';
            % Update bias
            hidden_bias = hidden_bias -learning_rate*s_hidden;
            output_bias = output_bias -learning_rate*s_output;
            % Back propagation end
        end
        % Calcualte mean square error
        avgCost = avgCost / k;
        arrayCost(j)=avgCost;
        avgCost = 0;
        arrayEpoch(j)=j;
    end
%     figure
%     hold on
% hold on
%             
% plot(arrayEpoch,arrayCost,'r')
% ylim([0 1])
% title('Mean square error versus number of epoch (80 neurons)')
% ylabel('Mean square error') 
% xlabel('Number of epoch') 

end