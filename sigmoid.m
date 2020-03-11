function transfer_output = sigmoid(input)
    for i = 1:size(input,1)
        transfer_output(i,1) = double(1/(1+exp(-1*input(i,1))));
    end
end
