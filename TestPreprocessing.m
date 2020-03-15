function TestPreprocessing()
 T = readtable('test.csv');
 fashion = table2array(T);
 f_normalized= normalize(fashion);
 [r,c]=size(f_normalized);
 tests = zeros(r,c-1);
 ids = zeros(r,1);
 for i = 1:r
     row = f_normalized(i,2:c);
     tests(i,:)=row;
     ids(i,1)= fashion(i,1);
 end 
save('fashion_test.mat','tests','ids');  
