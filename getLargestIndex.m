function [largestIndex] = getLargestIndex(digit) 
[r,c]=size(digit);
largestIndex=1;
for i = 1:r
if(digit(i,1)> digit(largestIndex)) 
    largestIndex=i;
end
end
