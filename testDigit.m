function [success]=testDigit(input, numb_noise, num_sample, targetIndex) 
    % Load weight and bias
    load('digitTrained.mat');
    % Total number of success
    success=0;
    for i = 1:num_sample
        p=addNoise(input,numb_noise);
        t=testInput(p',hidden_weights,final_weights,hidden_bias,final_bias); 
        if(getLargestIndex(t)==targetIndex)
            success=success+1; 
        end     
    end
end 