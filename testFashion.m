% FashionPreprocessing();
load('fashion.mat');
input_neurons = 784;
hidden_neurons = 100;
hidden_neurons2 = 50;
output_neurons = 10;
batch = 1;
learning_rate = 0.01;
epoch =50;
[weights, biases]=backProp_Multi(inputs,targets,input_neurons,hidden_neurons,hidden_neurons2,output_neurons,epoch, learning_rate);
save('fashionTrained.mat','weights','biases');  
% [weights, biases]=backProp_Batch(inputs,targets,batch,input_neurons,hidden_neurons,hidden_neurons2,output_neurons,epoch,learning_rate);
% save('fashionTrained.mat','weights','biases');  
load('fashionTrained.mat');
load('fashion_test.mat');
t=test_Multi(inputs(2,:)',2,weights, biases);
 testFSet(inputs,targets)
% 
% % testSingleSet(train3,3);
% % % % [ti]=testInput(inputs(5,:),hidden_weights,final_weights,hidden_bias,final_bias)
% % testSet(test0,1)
%  output = generateTest(tests);
% save('fashionTestRes.mat','output');
%  T = array2table(output')
%  T.Properties.VariableNames(1:2) = {'Id','label'}
%  writetable(T,'submit.csv');


