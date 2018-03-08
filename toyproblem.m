clear all; close all;

seedCount=2;
labelCount=2;
edgeCount=5;
pixelCount=6;
seeds=[1,6];
unseeds=1:6
unseeds(seeds)=[]
labels=[1,2];
edges=[1,2;2,3;3,4;4,5;5,6];
pixels=[1,2,3,4,5,6];
weights=[1,1,1,1,1];

A=zeros(edgeCount, pixelCount);
for i = 1:1:edgeCount
    A(i, edges(i,1))=-1;
    A(i, edges(i,2))=1;
end
A

C=zeros(edgeCount, edgeCount);
for i = 1:1:edgeCount
    C(i,i)=weights(i);
end
C
AtC=A'*C
Lap=AtC*A

Lu=Lap(unseeds,unseeds)
Lb=-Lap(unseeds, seeds)
b=zeros(seedCount, labelCount);
for i=1:1:seedCount
    b(i, labels(i))=1;
end
b
RHS = Lb*b

X=Lu\RHS


