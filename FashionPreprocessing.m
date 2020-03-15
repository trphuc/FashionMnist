function FashionPreprocessing()
 T = readtable('train.csv');
 fashion = table2array(T);
 f_normalized= normalize(fashion);
 [r,c]=size(fashion);
 train0 = zeros(6021, c-2);
 target0 = zeros(10, 6021);
 train1 = zeros(6002, c-2);
 target1 = zeros(10, 6002);
 train2 = zeros(5991, c-2);
 target2 = zeros(10, 5991);
 train3 = zeros(5981, c-2);
 target3 = zeros(10, 5981);
 train4 = zeros(5950, c-2);
 target4 = zeros(10, 5950);
 train5 = zeros(5970, c-2);
 target5 = zeros(10, 5970);
 train6 = zeros(6008, c-2);
 target6 = zeros(10, 6008);
 train7 = zeros(6029, c-2);
 target7 = zeros(10, 6029);
 train8 = zeros(6015, c-2);
 target8 = zeros(10, 6015);
 train9 = zeros(6033, c-2);
 target9 = zeros(10, 6033);
 inputs = zeros(r,c-2);
 targets = zeros(10,r);
 count0=1;
 count1=1;
 count2=1;
 count3=1;
 count4=1;
 count5=1;
 count6=1;
 count7=1;
 count8=1;
 count9=1;
 for i = 1:r
     row = f_normalized(i,3:c);
     target = fashion(i,2);
     inputs(i,:)=row;
     targets(:,i)= fashion_taget(target);
     if(target==0)
     train0(count0,:)=row;
     target0(:,i)= fashion_taget(0);
     count0=count0+1;
     end
     if(target==1)
     train1(count1,:)=row;
     target1(:,i)= fashion_taget(1);
     count1=count1+1;
     end
     if(target==2)
     train2(count2,:)=row;
     target2(:,i)= fashion_taget(2);
     count2=count2+1;
     end
     if(target==3)
     train3(count3,:)=row;
     target3(:,i)= fashion_taget(3);
     count3=count3+1;
     end
     if(target==4)
     train4(count4,:)=row;
     target4(:,i)= fashion_taget(4);
     count4=count4+1;
     end
     if(target==5)
     train5(count5,:)=row;
     target5(:,i)= fashion_taget(5);
     count5=count5+1;
     end
     if(target==6)
     train6(count6,:)=row;
     target6(:,i)= fashion_taget(6);
     count6=count6+1;
     end
     if(target==7)
     train7(count7,:)=row;
     target7(:,i)= fashion_taget(7);
     count7=count7+1;
     end
     if(target==8)
     train8(count8,:)=row;
     target8(:,i)= fashion_taget(8);
     count8=count8+1;
     end
     if(target==9)
     train9(count9,:)=row;
     target9(:,i)= fashion_taget(9);
     count9=count9+1;
     end
 end 
save('fashion.mat','inputs','targets','train0','train1','train2','train3','train4','train5','train6','train7','train8','train9','target0','target1','target2','target3','target4','target5','target6','target7','target8','target9'); 