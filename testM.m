input_neurons = 30;
hidden_neurons = 20;
hidden_neurons2 = 30;

output_neurons = 3;
hidden_layer=2;
lrate = 0.1;
loop = 1000;
p0=[-1 1 1 1 1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 -1 1 1 1 1 -1];
p1=[-1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1];
p2 = [1 -1 -1 -1 -1 -1 1 -1 -1 1 1 1 1 -1 -1 1 -1 1 -1 1 1 -1 -1 1 -1 -1 -1 -1 -1 1]; 
p3 = [-1 1 -1 -1 1 -1 1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 1 -1 -1 1 -1 1 1 1 1 1 1 1]; 

P = [p0;p1;p2];
t1=[1;0;0];
t2=[0;1;0];
t3=[0;0;1];

T=[t1 t2 t3];

[weights,biases] = backProp_Multi(P,T,2,input_neurons,hidden_neurons,hidden_neurons,output_neurons,loop, lrate);
% [hidden_weights,final_weights,hidden_bias,final_bias] = backProp(P,T,input_neurons,hidden_neurons,output_neurons,loop, lrate);

p=addNoise(p0,8);

[final_output] = test_Multi(p',hidden_layer,weights,biases)
% 
% [net_final] = testInput(p1,hidden_weights,final_weights,hidden_bias,final_bias)
% % weights_array = {hidden_weights,hidden_weights2,final_weights};
% % bias_array = {hidden_bias,hidden_bias2,final_bias};
% testDigit(p2',8,100,3)
