function [s] = sensitive(differ,final_output,final_hidden2,final_hidden,weights)
s{1,3}=[];
s{1,3} = -2*differ.* sigmoid_Prime(final_output); 
s{1,2} = sigmoid_Prime(final_hidden2).*(weights{3,1}'*s{1,3} );
s{1,1} = sigmoid_Prime(final_hidden).*(weights{2,1}'*s{1,2} );
end

