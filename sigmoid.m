function p = sigmoid(input)
    [r,c]=size(input);
    p=zeros(r,c);
    for i = 1:size(input,1)
        p(i,1) = double(1/(1+exp(-1*input(i,1))));
    end
end
