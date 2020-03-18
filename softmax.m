function [output] = softmax(input)
mx=max(input);
input=input-mx;
output=exp(input)/sum(exp(input));
end
