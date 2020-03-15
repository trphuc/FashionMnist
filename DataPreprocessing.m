function [inputs, targets]=DataPreprocessing()
    load('mnist_all.mat');
    [r0,c]=size(train5);
    r0=2;
    inputs = zeros(r0*10,784);
    targets = zeros(10,r0*10);
    % Only the smallest number of sample of all set to train
    offset=0;
    for k = 0:9
        target_example = [0;0;0;0;0;0;0;0;0;0];
        for i = 1 : r0
            switch k
                case 0
                    train_example = train0(i,:);
                case 1
                    train_example = train1(i,:);
                case 2
                    train_example = train2(i,:);
                case 3
                    train_example = train3(i,:);
                case 4
                    train_example = train4(i,:);
                case 5
                    train_example = train5(i,:);
                 case 6
                    train_example = train6(i,:);
                case 7
                    train_example = train7(i,:);
                case 8
                    train_example = train8(i,:);
                case 9
                    train_example = train9(i,:);
            end 
            inputs(i+offset,:) = train_example;
            target_example(k+1,1)=1;
            targets(:,i+offset) = target_example;
        end
        offset= offset+r0;
    end 
end 



