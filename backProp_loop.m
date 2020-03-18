function  [weights,biases]=backProp_loop(inputs,targets,batch,input_neurons,hidden_neurons,hidden_neurons2,output_neurons,epoch, learning_rate)
% Number of sample data
k = size(inputs,1);
num_batch = k/batch;
offset = 1;
neurons=[input_neurons,hidden_neurons,hidden_neurons2,output_neurons];
[weights,biases]=initWeights(2,neurons);
for i = 1:num_batch
    input=inputs(offset:i*batch,:);
    target = targets(:,offset:i*batch);
    offset = i*batch+1;
    [weights,biases]=backProp_batch(input,target,batch,neurons,weights,biases, epoch, learning_rate);
end
end