
input_neurons = 784;
hidden_neurons = 100;
output_neurons = 10;
[inputs, targets]=DataPreprocessing();
save('mnistProcessed.mat','inputs','targets');  
load('mnistProcessed.mat');
lrate = 0.01;
loop = 1000;
% % [hidden_weights,final_weights,hidden_bias,final_bias] = backPropMnist(inputs,targets, input_neurons,hidden_neurons,output_neurons,loop, lrate);
[weights, biases] = backProp_Multi(inputs,targets,2,input_neurons,100,50,output_neurons,loop, lrate);
save('mnistTrained.mat','weights','biases');  
% load('mnistTrained.mat');
% load('mnist_all.mat');
% % % [ti]=testInput(inputs(5,:),hidden_weights,final_weights,hidden_bias,final_bias)
% testSet(test0,1)

