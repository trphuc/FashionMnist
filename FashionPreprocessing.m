function FashionPreprocessing()
 T = readtable('train.csv');
 fashion = table2array(T);
 [r,c]=size(fashion);
 inputs = zeros(r,c-2);
 targets = zeros(10,r);
 for i = 1:r
     row = fashion(i,3:c);
     target = fashion(i,2);
     inputs(i,:)=row;
     target_example = [0;0;0;0;0;0;0;0;0;0];
     target_example(target+1,1)=1;
     targets(:,i)=target_example;
 end 
save('fashion.mat','inputs','targets');  
