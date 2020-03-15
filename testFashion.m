% FashionPreprocessing();
load('fashion.mat');
input_neurons = 784;
hidden_neurons = 100;
output_neurons = 10;
lrate = 0.01;
loop = 300;
[weights, biases] = backProp_Multi(inputs,targets,2,input_neurons,100,50,output_neurons,loop, lrate);
save('fashionTrained.mat','weights','biases');  
load('fashionTrained.mat');
load('fashion_test.mat');
% test_Multi(train8(2,:)',2,weights, biases)
% testFSet(inputs,targets)
% testSingleSet(train3,3);
% % % [ti]=testInput(inputs(5,:),hidden_weights,final_weights,hidden_bias,final_bias)
% testSet(test0,1)
% output = generateTest(tests);
% save('fashionTestRes.mat','output');
% T = array2table(output')
% T.Properties.VariableNames(1:2) = {'Id','label'}
% writetable(T,'submit.csv');
