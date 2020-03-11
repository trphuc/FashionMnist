% FashionPreprocessing();
load('fashion.mat');
input_neurons = 784;
hidden_neurons = 100;
output_neurons = 10;
lrate = 0.01;
loop = 10;
% % [hidden_weights,final_weights,hidden_bias,final_bias] = backPropMnist(inputs,targets, input_neurons,hidden_neurons,output_neurons,loop, lrate);
% [weights, biases] = backProp_Multi(inputs,targets,2,input_neurons,100,50,output_neurons,loop, lrate);
% save('fashionTrained.mat','weights','biases');  
load('fashionTrained.mat');
% load('mnist_all.mat');
inputs(1,:)
test_Multi(inputs(1,:)',2,weights, biases)
% % % [ti]=testInput(inputs(5,:),hidden_weights,final_weights,hidden_bias,final_bias)
% testSet(test0,1)