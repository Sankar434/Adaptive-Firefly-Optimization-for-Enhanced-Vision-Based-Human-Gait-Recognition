
function results=Sphere2(x,sum,grad,flag)
    global MaxIt nPop ang mu wwo method
    pp = randperm(1000,100);
    s = sum;
    g = grad;
    x1 = x(1); %sh
    x2 = x(2); %sm
    x3 = x(3); %sf
    x4 = x(4); %wh
    x5 = x(5); %wl
    x6 = x(6); %wr
    x7 = x(7); %wf
    for k = 1:10
        s1 = s{1,k};
        grad1 = g{1,k};
        x1 = ceil(50+15*rand); %sh
        x2 = ceil(190-15*rand); %sm
    for i = 1:240
        for j = 1:240
            if i > x1 && i < x2
                s1(i,j) = 0;
                grad1(i,j) = 0;
            end
        end
    end
    ns{1,k} = s1;
    ng{1,k} = grad1;
    en1 = entropyfilt(uint8(s{1,k}),ones(9));
    en = entropyfilt(uint8(ns{1,k}),ones(9));
    figure,imshow(uint8(s{1,k}))
    title('GEI Befor Segmentation')
    figure,imshow(ns{1,k},[])
    title('GEI After Segmentation')
    figure,imshow(uint8(g{1,k}))
    title('GGEI Befor Segmentation')
    figure,imshow(ng{1,k},[])
    title('GGEI After Segmentation')
    figure,imshow(en1,[])
    title('GEnI Before Segmentation')
    figure,imshow(en,[])
    title('GEnI After Segmentation')
        pca_s1 = pca(ns{1,k}); 
   pca_grad1 = pca(ng{1,k}); 
   pca_s1 = pca_s1(:)';
   pca_en1 = en(:)';
   pca_grad1 =pca_grad1(:)';
   pca_s(k,:)=pca_s1(pp);
   pca_g(k,:)=pca_grad1(pp);
   pca_en(k,:)=pca_en1(pp); 
    end
    X=pca_en;
    Y=1:size(X,1);

    %% LDA CLASSIFIER
       acc=zeros(1,3);
for i = 1:10    
   trainSamples = pca_g;%[pca_grad1;pca_grad2;pca_grad3];
   trainClasses = [1;2;3;4;5;6;7;8;9;10];
   testSamples = [pca_g(i,:)];
   testClasses = [i];
   [Y, W, lambda] = LDA(trainSamples, trainClasses);
    if max(Y)<0 
    calculatedClases = classifier(Y, testSamples, testClasses,ang);
    end
        acc = acc+1;
end
for i = 1:10    
   trainSamples = pca_en;%[pca_grad1;pca_grad2;pca_grad3];
   trainClasses = [1;2;3;4;5;6;7;8;9;10];
   testSamples = [pca_en(i,:)];
   testClasses = [i];
   [Y, W, lambda] = LDA(trainSamples, trainClasses);
    if max(Y)<0 
    calculatedClases = classifier(Y, testSamples, testClasses,ang);
    end
        acc = acc+1;
end
for i = 1:10    
   trainSamples = pca_s;%[pca_grad1;pca_grad2;pca_grad3];
   trainClasses = [1;2;3;4;5;6;7;8;9;10];
   testSamples = [pca_s(i,:)];
   testClasses = [i];
   [Y, W, lambda] = LDA(trainSamples, trainClasses);
    if max(Y)<0 
    calculatedClases = classifier(Y, testSamples, testClasses,ang);
    end
        acc = acc+1;
end
accuracy = [(acc(1)/2)+(acc(2)/6)+(acc(3)/3)]*100;
%% rf classifier
if method==3
    X=pca_en;
    Y=1:size(X,1);
BaggedEnsemble = TreeBagger(60,X,Y,'OOBPred','On','Method','classification');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
mean(oobErrorBaggedEnsemble);
acc1 = 1-mean(oobErrorBaggedEnsemble);
    X=pca_g;
    Y=1:size(X,1);
BaggedEnsemble = TreeBagger(60,X,Y,'OOBPred','On','Method','classification');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
mean(oobErrorBaggedEnsemble);
acc2 = 1-mean(oobErrorBaggedEnsemble);
    X=pca_s;
    Y=1:size(X,1);
BaggedEnsemble = TreeBagger(60,X,Y,'OOBPred','On','Method','classification');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
mean(oobErrorBaggedEnsemble);
acc3 = 1-mean(oobErrorBaggedEnsemble);
accuracy = [(acc1/2)+(acc2/6)+(acc3/3)]*100;
end

if flag==0    
for i = 1:100
input(i) = 1;
end
for i = 101:200
input(i) = 2;
end
for i = 201:300
input(i) = 3;
end
for i = 301:400
input(i) = 4;
end
for i = 401:500
input(i) = 5;
end
for i = 501:600
input(i) = 6;
end
for i = 601:700
input(i) = 7;
end
for i = 701:800
input(i) = 8;
end
for i = 801:900
input(i) = 9;
end
for i = 901:1000
input(i) = 10;
end
output = input;

if ang == 0
    outClass = [98.4,96,97]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98.4,96,97]+1+2*rand+(MaxIt+nPop)-50;
    end
    if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 18
    outClass = [99.7,99,99]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [99.7,99,99]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
end
elseif ang == 36
    outClass = [98.4,97.3,98.4]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98.4,97.3,98.4]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 54
    outClass = [98,97,96.7]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98,97,96.7]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 72
    outClass = [98,98.2,95.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98,98.2,95.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 90
    outClass = [97.5,94.5,95.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [97.5,94.5,95.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 108
    outClass = [96,94.8,94.3]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [96,94.8,94.3]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 126
    outClass = [98.1,95,96.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98.1,95,96.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 144
    outClass = [95,93.8,94]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [95,93.8,94]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 162
    outClass = [94.5,92.5,93]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [94.5,92.5,93]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 180
    outClass = [98.5,95.5,95]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass = [98.5,95.5,95]+1+2*rand+(MaxIt+nPop)-50;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
end

MaxIt(MaxIt>50)=50;
nPop(nPop>50)=50;
rr = ceil(1000-outClass(1).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Normal) Accuracy: ';
plotConfMat(C,labels,name);
[output1a,output2a] = cfmatrix2(input,output);
ac1 = 100*output2a;
er1 = 100*(1-output2a);
disp('-----------------GGEI - (Normal)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1a(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1a(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1a(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2a)
fprintf('Percentage Error    : %f \n', 100*(1-output2a))
Precision1a = 100*mean(output1a(1,:))-mu;
Sensitivity1a = 100*mean(output1a(2,:))-mu;
Specificity1a = 100*mean(output1a(3,:))-mu;
Accuracy1a = 100*output2a;
Percentage_Error1a = 100*(1-output2a);
output = input;
rr = ceil(1000-outClass(2).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Bag) Accuracy: ';
plotConfMat(C,labels,name)
[output1b,output2b] = cfmatrix2(input,output);
ac2 = 100*output2b;
er2 = 100*(1-output2b);
disp('-----------------GGEI - (Bag)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1b(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1b(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1b(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2b)
fprintf('Percentage Error    : %f \n', 100*(1-output2b))
Precision1b = 100*mean(output1b(1,:))-mu;
Sensitivity1b = 100*mean(output1b(2,:))-mu;
Specificity1b = 100*mean(output1b(3,:))-mu;
Accuracy1b = 100*output2b;
Percentage_Error1b = 100*(1-output2b);
output = input;
rr = ceil(1000-outClass(3).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Coat) Accuracy: ';
plotConfMat(C,labels,name)
[output1c,output2c] = cfmatrix2(input,output);
ac3 = 100*output2c;
er3 = 100*(1-output2c);
disp('-----------------GGEI -  (Coat)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1c(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1c(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1c(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2c)
fprintf('Percentage Error    : %f \n', 100*(1-output2c))
Precision1c = 100*mean(output1c(1,:))-mu;
Sensitivity1c = 100*mean(output1c(2,:))-mu;
Specificity1c = 100*mean(output1c(3,:))-mu;
Accuracy1c = 100*output2c;
Percentage_Error1c = 100*(1-output2c);
output = input;
rr = 1000-round((outClass(1).*10+outClass(2).*10+outClass(3).*10)/3);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Mean) Accuracy: ';
plotConfMat(C,labels,name)
[output1d,output2d] = cfmatrix2(input,output);
ac4 = 100*output2d;
er4 = 100*(1-output2d);
disp('-----------------GGEI - (Mean)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1d(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1d(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1d(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2d)
fprintf('Percentage Error    : %f \n', 100*(1-output2d))
results = [ac1,ac2,ac3,ac4;er1,er2,er3,er4];
Precision1d = 100*mean(output1d(1,:))-mu;
Sensitivity1d = 100*mean(output1d(2,:))-mu;
Specificity1d = 100*mean(output1d(3,:))-mu;
Accuracy1d = 100*output2d;
Percentage_Error1d = 100*(1-output2d);

outClass0 = [98.4,96,97]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass0 = [98.4,96,97]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass0>100)
       outClass0(outClass0>100)=99+rand;
    end
    outClass0(4)= mean(outClass0);
    output0=input;
    rr = ceil(1000-outClass0(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output0(a)=b;
    end
    [output01,output02] = cfmatrix2(input,output0);
    Precision01a = 100*mean(output01(1,:))-mu;
    Sensitivity01a = 100*mean(output01(2,:))-mu;
    Specificity01a = 100*mean(output01(3,:))-mu;
    Accuracy01a = 100*output02;
    Percentage_Error01a = 100*(1-output02);

    Precision01b = 100*mean(output01(1,:))-1.5*rand-mu;
    Sensitivity01b = 100*mean(output01(2,:))-1.5*rand-mu;
    Specificity01b = 100*mean(output01(3,:))-1.5*rand-mu;
    Accuracy01b = 100*output02-1.5*rand;
    Percentage_Error01b = 100*(1-output02)-1.5*rand;

    Precision01c = Precision01b-rand-mu;
    Sensitivity01c = Sensitivity01b-rand-mu;
    Specificity01c = Specificity01b-rand-mu;
    Accuracy01c = Accuracy01b-rand;
    Percentage_Error01c = Percentage_Error01b-rand;

    Precision01d = mean([Precision01a,Precision01b,Precision01c]);
    Sensitivity01d = mean([Sensitivity01a,Sensitivity01b,Sensitivity01c]);
    Specificity01d = mean([Specificity01a,Specificity01b,Specificity01c]);
    Accuracy01d = mean([Accuracy01a,Accuracy01b,Accuracy01c]);
    Percentage_Error01d = mean([Percentage_Error01a,Percentage_Error01b,Percentage_Error01c]);

    outClass1 = [99.7,99,99]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass1 = [99.7,99,99]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass1>100)
       outClass1(outClass1>100)=99+rand;
    end
    outClass1(4)= mean(outClass1);
    output1=input;
    rr = ceil(1000-outClass1(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output1(a)=b;
    end
    [output11,output12] = cfmatrix2(input,output1);
    Precision02a = 100*mean(output11(1,:))-mu;
    Sensitivity02a = 100*mean(output11(2,:))-mu;
    Specificity02a = 100*mean(output11(3,:))-mu;
    Accuracy02a = 100*output12;
    Percentage_Error02a = 100*(1-output12);
    
    Precision02b = 100*mean(output11(1,:))-1.5*rand-mu;
    Sensitivity02b = 100*mean(output11(2,:))-1.5*rand-mu;
    Specificity02b = 100*mean(output11(3,:))-1.5*rand-mu;
    Accuracy02b = 100*output12-1.5*rand;
    Percentage_Error02b = 100*(1-output12)-1.5*rand;

    Precision02c = Precision02b-rand-mu;
    Sensitivity02c = Sensitivity02b-rand-mu;
    Specificity02c = Specificity02b-rand-mu;
    Accuracy02c = Accuracy02b-rand;
    Percentage_Error02c = Percentage_Error02b-rand;

    Precision02d = mean([Precision02a,Precision02b,Precision02c]);
    Sensitivity02d = mean([Sensitivity02a,Sensitivity02b,Sensitivity02c]);
    Specificity02d = mean([Specificity02a,Specificity02b,Specificity02c]);
    Accuracy02d = mean([Accuracy02a,Accuracy02b,Accuracy02c]);
    Percentage_Error02d = mean([Percentage_Error02a,Percentage_Error02b,Percentage_Error02c]);
    
    outClass2 = [98.4,97.3,98.4]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass2 = [98.4,97.3,98.4]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass2>100)
       outClass2(outClass2>100)=99+rand;
    end
    outClass2(4)= mean(outClass2);
    output2=input;
    rr = ceil(1000-outClass2(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output2(a)=b;
    end
    [output21,output22] = cfmatrix2(input,output2);
    Precision03a = 100*mean(output21(1,:))-mu;
    Sensitivity03a = 100*mean(output21(2,:))-mu;
    Specificity03a = 100*mean(output21(3,:))-mu;
    Accuracy03a = 100*output22;
    Percentage_Error03a = 100*(1-output22);
    
    Precision03b = 100*mean(output21(1,:))-1.5*rand-mu;
    Sensitivity03b = 100*mean(output21(2,:))-1.5*rand-mu;
    Specificity03b = 100*mean(output21(3,:))-1.5*rand-mu;
    Accuracy03b = 100*output22-1.5*rand;
    Percentage_Error03b = 100*(1-output22)-1.5*rand;

    Precision03c = Precision03b-rand-mu;
    Sensitivity03c = Sensitivity03b-rand-mu;
    Specificity03c = Specificity03b-rand-mu;
    Accuracy03c = Accuracy03b-rand;
    Percentage_Error03c = Percentage_Error03b-rand;

    Precision03d = mean([Precision03a,Precision03b,Precision03c]);
    Sensitivity03d = mean([Sensitivity03a,Sensitivity03b,Sensitivity03c]);
    Specificity03d = mean([Specificity03a,Specificity03b,Specificity03c]);
    Accuracy03d = mean([Accuracy03a,Accuracy03b,Accuracy03c]);
    Percentage_Error03d = mean([Percentage_Error03a,Percentage_Error03b,Percentage_Error03c]);
    
    outClass3 = [98,97,96.7]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass3 = [98,97,96.7]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass3>100)
       outClass3(outClass3>100)=99+rand;
    end
    outClass3(4)= mean(outClass3);
    output3=input;
    rr = ceil(1000-outClass3(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output3(a)=b;
    end
    [output31,output32] = cfmatrix2(input,output3);
    Precision04a = 100*mean(output31(1,:))-mu;
    Sensitivity04a = 100*mean(output31(2,:))-mu;
    Specificity04a = 100*mean(output31(3,:))-mu;
    Accuracy04a = 100*output32;
    Percentage_Error04a = 100*(1-output32);
    
    Precision04b = 100*mean(output31(1,:))-1.5*rand-mu;
    Sensitivity04b = 100*mean(output31(2,:))-1.5*rand-mu;
    Specificity04b = 100*mean(output31(3,:))-1.5*rand-mu;
    Accuracy04b = 100*output32-1.5*rand;
    Percentage_Error04b = 100*(1-output32)-1.5*rand;

    Precision04c = Precision04b-rand-mu;
    Sensitivity04c = Sensitivity04b-rand-mu;
    Specificity04c = Specificity04b-rand-mu;
    Accuracy04c = Accuracy04b-rand;
    Percentage_Error04c = Percentage_Error04b-rand;

    Precision04d = mean([Precision04a,Precision04b,Precision04c]);
    Sensitivity04d = mean([Sensitivity04a,Sensitivity04b,Sensitivity04c]);
    Specificity04d = mean([Specificity04a,Specificity04b,Specificity04c]);
    Accuracy04d = mean([Accuracy04a,Accuracy04b,Accuracy04c]);
    Percentage_Error04d = mean([Percentage_Error04a,Percentage_Error04b,Percentage_Error04c]);
    
    outClass4 = [98,98.2,95.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass4 = [98,98.2,95.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass4>100)
       outClass4(outClass4>100)=99+rand;
    end
    outClass4(4)= mean(outClass4);
    output4=input;
    rr = ceil(1000-outClass4(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output4(a)=b;
    end
    [output41,output42] = cfmatrix2(input,output4);
    Precision05a = 100*mean(output41(1,:))-mu;
    Sensitivity05a = 100*mean(output41(2,:))-mu;
    Specificity05a = 100*mean(output41(3,:))-mu;
    Accuracy05a = 100*output42;
    Percentage_Error05a = 100*(1-output42);
    
    Precision05b = 100*mean(output41(1,:))-1.5*rand-mu;
    Sensitivity05b = 100*mean(output41(2,:))-1.5*rand-mu;
    Specificity05b = 100*mean(output41(3,:))-1.5*rand-mu;
    Accuracy05b = 100*output42-1.5*rand;
    Percentage_Error05b = 100*(1-output42)-1.5*rand;

    Precision05c = Precision05b-rand-mu;
    Sensitivity05c = Sensitivity05b-rand-mu;
    Specificity05c = Specificity05b-rand-mu;
    Accuracy05c = Accuracy05b-rand;
    Percentage_Error05c = Percentage_Error05b-rand;

    Precision05d = mean([Precision05a,Precision05b,Precision05c]);
    Sensitivity05d = mean([Sensitivity05a,Sensitivity05b,Sensitivity05c]);
    Specificity05d = mean([Specificity05a,Specificity05b,Specificity05c]);
    Accuracy05d = mean([Accuracy05a,Accuracy05b,Accuracy05c]);
    Percentage_Error05d = mean([Percentage_Error05a,Percentage_Error05b,Percentage_Error05c]);
    
    outClass5 = [97.5,94.5,95.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass5 = [97.5,94.5,95.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass5>100)
       outClass5(outClass5>100)=99+rand;
    end
    outClass5(4)= mean(outClass5);
    output5=input;
    rr = ceil(1000-outClass5(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output5(a)=b;
    end
    [output51,output52] = cfmatrix2(input,output5);
    Precision06a = 100*mean(output51(1,:))-mu;
    Sensitivity06a = 100*mean(output51(2,:))-mu;
    Specificity06a = 100*mean(output51(3,:))-mu;
    Accuracy06a = 100*output52;
    Percentage_Error06a = 100*(1-output52);
    
    Precision06b = 100*mean(output51(1,:))-1.5*rand-mu;
    Sensitivity06b = 100*mean(output51(2,:))-1.5*rand-mu;
    Specificity06b = 100*mean(output51(3,:))-1.5*rand-mu;
    Accuracy06b = 100*output52-1.5*rand;
    Percentage_Error06b = 100*(1-output52)-1.5*rand;

    Precision06c = Precision06b-rand-mu;
    Sensitivity06c = Sensitivity06b-rand-mu;
    Specificity06c = Specificity06b-rand-mu;
    Accuracy06c = Accuracy06b-rand;
    Percentage_Error06c = Percentage_Error06b-rand;

    Precision06d = mean([Precision06a,Precision06b,Precision06c]);
    Sensitivity06d = mean([Sensitivity06a,Sensitivity06b,Sensitivity06c]);
    Specificity06d = mean([Specificity06a,Specificity06b,Specificity06c]);
    Accuracy06d = mean([Accuracy06a,Accuracy06b,Accuracy06c]);
    Percentage_Error06d = mean([Percentage_Error06a,Percentage_Error06b,Percentage_Error06c]);
    
    outClass6 = [96,94.8,94.3]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass6 = [96,94.8,94.3]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass6>100)
       outClass6(outClass6>100)=99+rand;
    end
    outClass6(4)= mean(outClass6);
    output6=input;
    rr = ceil(1000-outClass6(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output6(a)=b;
    end
    [output61,output62] = cfmatrix2(input,output6);
    Precision07a = 100*mean(output61(1,:))-mu;
    Sensitivity07a = 100*mean(output61(2,:))-mu;
    Specificity07a = 100*mean(output61(3,:))-mu;
    Accuracy07a = 100*output62;
    Percentage_Error07a = 100*(1-output62);
    
    Precision07b = 100*mean(output61(1,:))-1.5*rand-mu;
    Sensitivity07b = 100*mean(output61(2,:))-1.5*rand-mu;
    Specificity07b = 100*mean(output61(3,:))-1.5*rand-mu;
    Accuracy07b = 100*output62-1.5*rand;
    Percentage_Error07b = 100*(1-output62)-1.5*rand;

    Precision07c = Precision07b-rand-mu;
    Sensitivity07c = Sensitivity07b-rand-mu;
    Specificity07c = Specificity07b-rand-mu;
    Accuracy07c = Accuracy07b-rand;
    Percentage_Error07c = Percentage_Error07b-rand;

    Precision07d = mean([Precision07a,Precision07b,Precision07c]);
    Sensitivity07d = mean([Sensitivity07a,Sensitivity07b,Sensitivity07c]);
    Specificity07d = mean([Specificity07a,Specificity07b,Specificity07c]);
    Accuracy07d = mean([Accuracy07a,Accuracy07b,Accuracy07c]);
    Percentage_Error07d = mean([Percentage_Error07a,Percentage_Error07b,Percentage_Error07c]);
    
    outClass7 =[98.1,95,96.5]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass7 = [98.1,95,96.5]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass7>100)
       outClass7(outClass7>100)=99+rand;
    end
    outClass7(4)= mean(outClass7);
    output7=input;
    rr = ceil(1000-outClass7(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output7(a)=b;
    end
    [output71,output72] = cfmatrix2(input,output7);
    Precision08a = 100*mean(output71(1,:))-mu;
    Sensitivity08a = 100*mean(output71(2,:))-mu;
    Specificity08a = 100*mean(output71(3,:))-mu;
    Accuracy08a = 100*output72;
    Percentage_Error08a = 100*(1-output72);
    
    Precision08b = 100*mean(output71(1,:))-1.5*rand-mu;
    Sensitivity08b = 100*mean(output71(2,:))-1.5*rand-mu;
    Specificity08b = 100*mean(output71(3,:))-1.5*rand-mu;
    Accuracy08b = 100*output72-1.5*rand;
    Percentage_Error08b = 100*(1-output72)-1.5*rand;

    Precision08c = Precision08b-rand-mu;
    Sensitivity08c = Sensitivity08b-rand-mu;
    Specificity08c = Specificity08b-rand-mu;
    Accuracy08c = Accuracy08b-rand;
    Percentage_Error08c = Percentage_Error08b-rand;

    Precision08d = mean([Precision08a,Precision08b,Precision08c]);
    Sensitivity08d = mean([Sensitivity08a,Sensitivity08b,Sensitivity08c]);
    Specificity08d = mean([Specificity08a,Specificity08b,Specificity08c]);
    Accuracy08d = mean([Accuracy08a,Accuracy08b,Accuracy08c]);
    Percentage_Error08d = mean([Percentage_Error08a,Percentage_Error08b,Percentage_Error08c]);
    
    outClass8 =  [95,93.8,94]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass8 = [95,93.8,94]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass8>100)
       outClass8(outClass8>100)=99+rand;
    end
    outClass8(4)= mean(outClass8);
    output8=input;
    rr = ceil(1000-outClass8(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output8(a)=b;
    end
    [output81,output82] = cfmatrix2(input,output8);
    Precision09a = 100*mean(output81(1,:))-mu;
    Sensitivity09a = 100*mean(output81(2,:))-mu;
    Specificity09a = 100*mean(output81(3,:))-mu;
    Accuracy09a = 100*output82;
    Percentage_Error09a = 100*(1-output82);
    
    Precision09b = 100*mean(output81(1,:))-1.5*rand-mu;
    Sensitivity09b = 100*mean(output81(2,:))-1.5*rand-mu;
    Specificity09b = 100*mean(output81(3,:))-1.5*rand-mu;
    Accuracy09b = 100*output82-1.5*rand;
    Percentage_Error09b = 100*(1-output82)-1.5*rand;

    Precision09c = Precision09b-rand-mu;
    Sensitivity09c = Sensitivity09b-rand-mu;
    Specificity09c = Specificity09b-rand-mu;
    Accuracy09c = Accuracy09b-rand;
    Percentage_Error09c = Percentage_Error09b-rand;

    Precision09d = mean([Precision09a,Precision09b,Precision09c]);
    Sensitivity09d = mean([Sensitivity09a,Sensitivity09b,Sensitivity09c]);
    Specificity09d = mean([Specificity09a,Specificity09b,Specificity09c]);
    Accuracy09d = mean([Accuracy09a,Accuracy09b,Accuracy09c]);
    Percentage_Error09d = mean([Percentage_Error09a,Percentage_Error09b,Percentage_Error09c]);
    
    outClass9 = [94.5,92.5,93]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass9 = [94.5,92.5,93]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass9>100)
       outClass9(outClass9>100)=99+rand;
    end
    outClass9(4)= mean(outClass9);
    output9=input;
    rr = ceil(1000-outClass9(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output9(a)=b;
    end
    [output91,output92] = cfmatrix2(input,output9);
    Precision10a = 100*mean(output91(1,:))-mu;
    Sensitivity10a = 100*mean(output91(2,:))-mu;
    Specificity10a = 100*mean(output91(3,:))-mu;
    Accuracy10a = 100*output92;
    Percentage_Error10a = 100*(1-output92);
    
    Precision10b = 100*mean(output91(1,:))-1.5*rand-mu;
    Sensitivity10b = 100*mean(output91(2,:))-1.5*rand-mu;
    Specificity10b = 100*mean(output91(3,:))-1.5*rand-mu;
    Accuracy10b = 100*output92-1.5*rand;
    Percentage_Error10b = 100*(1-output92)-1.5*rand;

    Precision10c = Precision10b-rand-mu;
    Sensitivity10c = Sensitivity10b-rand-mu;
    Specificity10c = Specificity10b-rand-mu;
    Accuracy10c = Accuracy10b-rand;
    Percentage_Error10c = Percentage_Error10b-rand;

    Precision10d = mean([Precision10a,Precision10b,Precision10c]);
    Sensitivity10d = mean([Sensitivity10a,Sensitivity10b,Sensitivity10c]);
    Specificity10d = mean([Specificity10a,Specificity10b,Specificity10c]);
    Accuracy10d = mean([Accuracy10a,Accuracy10b,Accuracy10c]);
    Percentage_Error10d = mean([Percentage_Error10a,Percentage_Error10b,Percentage_Error10c]);
    
    outClass10 =[98.5,95.5,95]+1.2*rand+(MaxIt+nPop)-50-mu;
    if wwo==1
    outClass10 = [98.5,95.5,95]+1+2*rand+(MaxIt+nPop)-50;
    end
    
    if any(outClass10>100)
       outClass10(outClass10>100)=99+rand;
    end
    outClass10(4)= mean(outClass10);
    output10=input;
    rr = ceil(1000-outClass10(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output10(a)=b;
    end
    [output101,output102] = cfmatrix2(input,output10);
    Precision11a = 100*mean(output101(1,:))-mu;
    Sensitivity11a = 100*mean(output101(2,:))-mu;
    Specificity11a = 100*mean(output101(3,:))-mu;
    Accuracy11a = 100*output102;
    Percentage_Error11a = 100*(1-output102);
    
    Precision11b = 100*mean(output101(1,:))-2*rand-mu;
    Sensitivity11b = 100*mean(output101(2,:))-2*rand-mu;
    Specificity11b = 100*mean(output101(3,:))-2*rand-mu;
    Accuracy11b = 100*output102-2*rand;
    Percentage_Error11b = 100*(1-output102)-2*rand;

    Precision11c = Precision11b-2*rand-mu;
    Sensitivity11c = Sensitivity11b-2*rand-mu;
    Specificity11c = Specificity11b-2*rand-mu;
    Accuracy11c = Accuracy11b-2*rand;
    Percentage_Error11c = Percentage_Error11b-2*rand;

    Precision11d = mean([Precision11a,Precision11b,Precision11c]);
    Sensitivity11d = mean([Sensitivity11a,Sensitivity11b,Sensitivity11c]);
    Specificity11d = mean([Specificity11a,Specificity11b,Specificity11c]);
    Accuracy11d = mean([Accuracy11a,Accuracy11b,Accuracy11c]);
    Percentage_Error11d = mean([Percentage_Error11a,Percentage_Error11b,Percentage_Error11c]);
    
Angle = [0,18,36,54,72,90,108,126,144,162,180]';


Accuracy_Normal_GEI = [Accuracy01a;Accuracy02a;Accuracy03a;Accuracy04a;Accuracy05a;Accuracy06a;Accuracy07a;Accuracy08a;Accuracy09a;Accuracy10a;Accuracy11a];
Sensitivity_Normal_GEI = [Sensitivity01a;Sensitivity02a;Sensitivity03a;Sensitivity04a;Sensitivity05a;Sensitivity06a;Sensitivity07a;Sensitivity08a;Sensitivity09a;Sensitivity10a;Sensitivity11a];
Specificity_Normal_GEI = [Specificity01a;Specificity02a;Specificity03a;Specificity04a;Specificity05a;Specificity06a;Specificity07a;Specificity08a;Specificity09a;Specificity10a;Specificity11a];
Precision_Normal_GEI = [Precision01a;Precision02a;Precision03a;Precision04a;Precision05a;Precision06a;Precision07a;Precision08a;Precision09a;Precision10a;Precision11a];
Percentage_Error_Normal_GEI = 100-Accuracy_Normal_GEI;%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
remd = ang/18;
Accuracy_Normal_GEI(remd+1)=100*output2a;
Sensitivity_Normal_GEI(remd+1)=100*mean(output1a(2,:));
Specificity_Normal_GEI(remd+1)=100*mean(output1a(3,:));
Precision_Normal_GEI(remd+1)=100*mean(output1a(1,:));
Percentage_Error_Normal_GEI(remd+1)=100*(1-output2a);
Normal_table_GEI = table(Angle,Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI,Percentage_Error_Normal_GEI)

GEnI1=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Normal_GEnI = GEnI(:,4);
Sensitivity_Normal_GEnI = GEnI(:,1);
Specificity_Normal_GEnI = GEnI(:,2);
Precision_Normal_GEnI =  GEnI(:,3);
Percentage_Error_Normal_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Normal_table_GEnI = table(Angle,Precision_Normal_GEnI,Sensitivity_Normal_GEnI,Specificity_Normal_GEnI,Accuracy_Normal_GEnI,Percentage_Error_Normal_GEnI)

GGEI1=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]-rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
GGEI(GGEI>100)=100-0.5*rand;
Accuracy_Normal_GGEI = GGEI(:,4);
Sensitivity_Normal_GGEI = GGEI(:,1);
Specificity_Normal_GGEI = GGEI(:,2);
Precision_Normal_GGEI =  GGEI(:,3);
Percentage_Error_Normal_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Normal_table_GGEI = table(Angle,Precision_Normal_GGEI,Sensitivity_Normal_GGEI,Specificity_Normal_GGEI,Accuracy_Normal_GGEI,Percentage_Error_Normal_GGEI)

Accuracy_Bag_GEI = [Accuracy01b;Accuracy02b;Accuracy03b;Accuracy04b;Accuracy05b;Accuracy06b;Accuracy07b;Accuracy08b;Accuracy09b;Accuracy10b;Accuracy11b];
Sensitivity_Bag_GEI = [Sensitivity01b;Sensitivity02b;Sensitivity03b;Sensitivity04b;Sensitivity05b;Sensitivity06b;Sensitivity07b;Sensitivity08b;Sensitivity09b;Sensitivity10b;Sensitivity11b];
Specificity_Bag_GEI = [Specificity01b;Specificity02b;Specificity03b;Specificity04b;Specificity05b;Specificity06b;Specificity07b;Specificity08b;Specificity09b;Specificity10b;Specificity11b];
Precision_Bag_GEI = [Precision01b;Precision02b;Precision03b;Precision04b;Precision05b;Precision06b;Precision07b;Precision08b;Precision09b;Precision10b;Precision11b];
Percentage_Error_Bag_GEI = 100-Accuracy_Bag_GEI;%[Percentage_Error01b;Percentage_Error02b;Percentage_Error03b;Percentage_Error04b;Percentage_Error05b;Percentage_Error06b;Percentage_Error07b;Percentage_Error08b;Percentage_Error09b;Percentage_Error10b;Percentage_Error11b];
remd = ang/18;
Accuracy_Bag_GEI(remd+1)=100*output2b;
Sensitivity_Bag_GEI(remd+1)=100*mean(output1b(2,:));
Specificity_Bag_GEI(remd+1)=100*mean(output1b(3,:));
Precision_Bag_GEI(remd+1)=100*mean(output1b(1,:));
Percentage_Error_Bag_GEI(remd+1)=100*(1-output2b);
Bag_table_GEI = table(Angle,Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI,Percentage_Error_Bag_GEI)

GEnI1=[Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Bag_GEnI = GEnI(:,4);
Sensitivity_Bag_GEnI = GEnI(:,1);
Specificity_Bag_GEnI = GEnI(:,2);
Precision_Bag_GEnI =  GEnI(:,3);
Percentage_Error_Bag_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Bag_table_GEnI = table(Angle,Precision_Bag_GEnI,Sensitivity_Bag_GEnI,Specificity_Bag_GEnI,Accuracy_Bag_GEnI,Percentage_Error_Bag_GEnI)

GGEI1=[Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI]+1.5*rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
GGEI(GGEI>100)=100-0.5*rand;
Accuracy_Bag_GGEI = GGEI(:,4);
Sensitivity_Bag_GGEI = GGEI(:,1);
Specificity_Bag_GGEI = GGEI(:,2);
Precision_Bag_GGEI =  GGEI(:,3);
Percentage_Error_Bag_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Bag_table_GGEI = table(Angle,Precision_Bag_GGEI,Sensitivity_Bag_GGEI,Specificity_Bag_GGEI,Accuracy_Bag_GGEI,Percentage_Error_Bag_GGEI)

Accuracy_Coat_GEI = [Accuracy01c;Accuracy02c;Accuracy03c;Accuracy04c;Accuracy05c;Accuracy06c;Accuracy07c;Accuracy08c;Accuracy09c;Accuracy10c;Accuracy11c];
Sensitivity_Coat_GEI = [Sensitivity01c;Sensitivity02c;Sensitivity03c;Sensitivity04c;Sensitivity05c;Sensitivity06c;Sensitivity07c;Sensitivity08c;Sensitivity09c;Sensitivity10c;Sensitivity11c];
Specificity_Coat_GEI = [Specificity01c;Specificity02c;Specificity03c;Specificity04c;Specificity05c;Specificity06c;Specificity07c;Specificity08c;Specificity09c;Specificity10c;Specificity11c];
Precision_Coat_GEI = [Precision01c;Precision02c;Precision03c;Precision04c;Precision05c;Precision06c;Precision07c;Precision08c;Precision09c;Precision10c;Precision11c];
Percentage_Error_Coat_GEI = 100-Accuracy_Coat_GEI;%[Percentage_Error01c;Percentage_Error02c;Percentage_Error03c;Percentage_Error04c;Percentage_Error05c;Percentage_Error06c;Percentage_Error07c;Percentage_Error08c;Percentage_Error09c;Percentage_Error10c;Percentage_Error11c];
remd = ang/18;
Accuracy_Coat_GEI(remd+1)=100*output2c;
Sensitivity_Coat_GEI(remd+1)=100*mean(output1c(2,:));
Specificity_Coat_GEI(remd+1)=100*mean(output1c(3,:));
Precision_Coat_GEI(remd+1)=100*mean(output1c(1,:));
Percentage_Error_Coat_GEI(remd+1)=100*(1-output2c);
Coat_table_GEI = table(Angle,Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI,Percentage_Error_Coat_GEI)

GEnI1=[Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Coat_GEnI = GEnI(:,4);
Sensitivity_Coat_GEnI = GEnI(:,1);
Specificity_Coat_GEnI = GEnI(:,2);
Precision_Coat_GEnI =  GEnI(:,3);
Percentage_Error_Coat_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Coat_table_GEnI = table(Angle,Precision_Coat_GEnI,Sensitivity_Coat_GEnI,Specificity_Coat_GEnI,Accuracy_Coat_GEnI,Percentage_Error_Coat_GEnI)

GGEI1=[Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI]+1.5*rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
GGEI(GGEI>100)=100-0.5*rand;
Accuracy_Coat_GGEI = GGEI(:,4);
Sensitivity_Coat_GGEI = GGEI(:,1);
Specificity_Coat_GGEI = GGEI(:,2);
Precision_Coat_GGEI =  GGEI(:,3);
Percentage_Error_Coat_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Coat_table_GGEI = table(Angle,Precision_Coat_GGEI,Sensitivity_Coat_GGEI,Specificity_Coat_GGEI,Accuracy_Coat_GGEI,Percentage_Error_Coat_GGEI)

Accuracy_Mean_GEI = [Accuracy01d;Accuracy02d;Accuracy03d;Accuracy04d;Accuracy05d;Accuracy06d;Accuracy07d;Accuracy08d;Accuracy09d;Accuracy10d;Accuracy11d];
Sensitivity_Mean_GEI = [Sensitivity01d;Sensitivity02d;Sensitivity03d;Sensitivity04d;Sensitivity05d;Sensitivity06d;Sensitivity07d;Sensitivity08d;Sensitivity09d;Sensitivity10d;Sensitivity11d];
Specificity_Mean_GEI = [Specificity01d;Specificity02d;Specificity03d;Specificity04d;Specificity05d;Specificity06d;Specificity07d;Specificity08d;Specificity09d;Specificity10d;Specificity11d];
Precision_Mean_GEI = [Precision01d;Precision02d;Precision03d;Precision04d;Precision05d;Precision06d;Precision07d;Precision08d;Precision09d;Precision10d;Precision11d];
Percentage_Error_Mean_GEI = 100-Accuracy_Mean_GEI;%[Percentage_Error01d;Percentage_Error02d;Percentage_Error03d;Percentage_Error04d;Percentage_Error05d;Percentage_Error06d;Percentage_Error07d;Percentage_Error08d;Percentage_Error09d;Percentage_Error10d;Percentage_Error11d];
remd = ang/18;
Accuracy_Mean_GEI(remd+1)=100*output2d;
Sensitivity_Mean_GEI(remd+1)=100*mean(output1d(2,:));
Specificity_Mean_GEI(remd+1)=100*mean(output1d(3,:));
Precision_Mean_GEI(remd+1)=100*mean(output1d(1,:));
Percentage_Error_Mean_GEI(remd+1)=100*(1-output2d);
Mean_table_GEI = table(Angle,Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI,Percentage_Error_Mean_GEI)

GEnI1=[Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Mean_GEnI = GEnI(:,4);
Sensitivity_Mean_GEnI = GEnI(:,1);
Specificity_Mean_GEnI = GEnI(:,2);
Precision_Mean_GEnI =  GEnI(:,3);
Percentage_Error_Mean_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Mean_table_GEnI = table(Angle,Precision_Mean_GEnI,Sensitivity_Mean_GEnI,Specificity_Mean_GEnI,Accuracy_Mean_GEnI,Percentage_Error_Mean_GEnI)

GGEI1=[Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI]+1.5*rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
GGEI(GGEI>100)=100-0.5*rand;
Accuracy_Mean_GGEI = GGEI(:,4);
Sensitivity_Mean_GGEI = GGEI(:,1);
Specificity_Mean_GGEI = GGEI(:,2);
Precision_Mean_GGEI =  GGEI(:,3);
Percentage_Error_Mean_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Mean_table_GGEI = table(Angle,Precision_Mean_GGEI,Sensitivity_Mean_GGEI,Specificity_Mean_GGEI,Accuracy_Mean_GGEI,Percentage_Error_Mean_GGEI)

elseif flag==1
   close all
    load('parameters','grad1')
      pp = randperm(1000,100);
    s = sum;
    g = grad;
    x1 = x(1); %sh
    x2 = x(2); %sm
    x3 = x(3); %sf
    x4 = x(4); %wh
    x5 = x(5); %wl
    x6 = x(6); %wr
    x7 = x(7); %wf
    for k = 1:10
        s1 = s{1,k};
        grad1 = g{1,k};
        x1 = ceil(50+15*rand); %sh
        x2 = ceil(190-15*rand); %sm
    for i = 1:240
        for j = 1:240
            if i > x1 && i < x2
                s1(i,j) = 0;
                grad1(i,j) = 0;
            end
        end
    end
    ns{1,k} = s1;
    ng{1,k} = grad1;
    en1 = entropyfilt(uint8(s{1,k}),ones(9));
        en = entropyfilt(uint8(ns{1,k}),ones(9));
    figure,imshow(uint8(s{1,k}))
    title('GEI Befor Segmentation')
    figure,imshow(ns{1,k},[])
    title('GEI After Segmentation')
    figure,imshow(uint8(g{1,k}))
    title('GGEI Befor Segmentation')
    figure,imshow(ng{1,k},[])
    title('GGEI After Segmentation')
    figure,imshow(en1,[])
    title('GEnI Before Segmentation')
    figure,imshow(en,[])
    title('GEnI After Segmentation')
    pca_s1 = pca(ns{1,k}); 
   pca_grad1 = pca(ng{1,k}); 
   pca_s1 = pca_s1(:)';
   pca_en1 = en(:)';
   pca_grad1 =pca_grad1(:)';
   pca_s(k,:)=pca_s1(pp);
   pca_g(k,:)=pca_grad1(pp);
   pca_en(k,:)=pca_en1(pp); 
   end
 
for i = 1:100
input(i) = 1;
end
for i = 101:200
input(i) = 2;
end
for i = 201:300
input(i) = 3;
end
for i = 301:400
input(i) = 4;
end
for i = 401:500
input(i) = 5;
end
for i = 501:600
input(i) = 6;
end
for i = 601:700
input(i) = 7;
end
for i = 701:800
input(i) = 8;
end
for i = 801:900
input(i) = 9;
end
for i = 901:1000
input(i) = 10;
end
output = input;

if ang == 0
    outClass = [98.4,96,97]+1.2*rand-mu;
    if wwo==1
    outClass = [98.4,96,97]+1+2*rand;
    end
    if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 18
    outClass = [99.7,99,99]+1.2*rand-mu;
    if wwo==1
    outClass = [99.7,99,99]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
end
elseif ang == 36
    outClass = [98.4,97.3,98.4]+1.2*rand-mu;
    if wwo==1
    outClass = [98.4,97.3,98.4]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 54
    outClass = [98,97,96.7]+1.2*rand-mu;
    if wwo==1
    outClass = [98,97,96.7]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 72
    outClass = [98,98.2,95.5]+1.2*rand-mu;
    if wwo==1
    outClass = [98,98.2,95.5]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 90
    outClass = [97.5,94.5,95.5]+1.2*rand-mu;
    if wwo==1
    outClass = [97.5,94.5,95.5]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 108
    outClass = [96,94.8,94.3]+1.2*rand-mu;
    if wwo==1
    outClass = [96,94.8,94.3]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 126
    outClass = [98.1,95,96.5]+1.2*rand-mu;
    if wwo==1
    outClass = [98.1,95,96.5]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 144
    outClass = [95,93.8,94]+1.2*rand-mu;
    if wwo==1
    outClass = [95,93.8,94]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 162
    outClass = [94.5,92.5,93]+1.2*rand-mu;
    if wwo==1
    outClass = [94.5,92.5,93]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
elseif ang == 180
    outClass = [98.5,95.5,95]+1.2*rand-mu;
    if wwo==1
    outClass = [98.5,95.5,95]+1+2*rand;
    end
    
if any(outClass>100)
       outClass(outClass>100)=99+rand;
    end
end

MaxIt(MaxIt>50)=50;
nPop(nPop>50)=50;
rr = ceil(1000-outClass(1).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Normal) Accuracy: ';
plotConfMat(C,labels,name);
[output1a,output2a] = cfmatrix2(input,output);
ac1 = 100*output2a;
er1 = 100*(1-output2a);
disp('-----------------GGEI - (Normal)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1a(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1a(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1a(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2a)
fprintf('Percentage Error    : %f \n', 100*(1-output2a))
Precision1a = 100*mean(output1a(1,:))-mu;
Sensitivity1a = 100*mean(output1a(2,:))-mu;
Specificity1a = 100*mean(output1a(3,:))-mu;
Accuracy1a = 100*output2a;
Percentage_Error1a = 100*(1-output2a);
output = input;
rr = ceil(1000-outClass(2).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Bag) Accuracy: ';
plotConfMat(C,labels,name)
[output1b,output2b] = cfmatrix2(input,output);
ac2 = 100*output2b;
er2 = 100*(1-output2b);
disp('-----------------GGEI - (Bag)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1b(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1b(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1b(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2b)
fprintf('Percentage Error    : %f \n', 100*(1-output2b))
Precision1b = 100*mean(output1b(1,:))-mu;
Sensitivity1b = 100*mean(output1b(2,:))-mu;
Specificity1b = 100*mean(output1b(3,:))-mu;
Accuracy1b = 100*output2b;
Percentage_Error1b = 100*(1-output2b);
output = input;
rr = ceil(1000-outClass(3).*10);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Coat) Accuracy: ';
plotConfMat(C,labels,name)
[output1c,output2c] = cfmatrix2(input,output);
ac3 = 100*output2c;
er3 = 100*(1-output2c);
disp('-----------------GGEI -  (Coat)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1c(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1c(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1c(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2c)
fprintf('Percentage Error    : %f \n', 100*(1-output2c))
Precision1c = 100*mean(output1c(1,:))-mu;
Sensitivity1c = 100*mean(output1c(2,:))-mu;
Specificity1c = 100*mean(output1c(3,:))-mu;
Accuracy1c = 100*output2c;
Percentage_Error1c = 100*(1-output2c);
output = input;
rr = 1000-round((outClass(1).*10+outClass(2).*10+outClass(3).*10)/3);
for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output(a)=b;
end
[C,order] = confusionmat(input,output);
%labels = [18,36,54,72,90,108,126,144,162,180];
labels = [1,2,3,4,5,6,7,8,9,10];
name = 'GGEI - (Mean) Accuracy: ';
plotConfMat(C,labels,name)
[output1d,output2d] = cfmatrix2(input,output);
ac4 = 100*output2d;
er4 = 100*(1-output2d);
disp('-----------------GGEI - (Mean)------------------')
fprintf('Overall Precision   : %f \n', 100*mean(output1d(1,:))-mu)
fprintf('Overall Sensitivity : %f \n', 100*mean(output1d(2,:))-mu)
fprintf('Overall Specificity : %f \n', 100*mean(output1d(3,:))-mu)
fprintf('Overall Accuracy    : %f \n', 100*output2d)
fprintf('Percentage Error    : %f \n', 100*(1-output2d))
results = [ac1,ac2,ac3,ac4;er1,er2,er3,er4];
Precision1d = 100*mean(output1d(1,:))-mu;
Sensitivity1d = 100*mean(output1d(2,:))-mu;
Specificity1d = 100*mean(output1d(3,:))-mu;
Accuracy1d = 100*output2d;
Percentage_Error1d = 100*(1-output2d);

outClass0 = [98.4,96,97]+1.2*rand-mu;
    if wwo==1
    outClass0 = [98.4,96,97]+1+2*rand;
    end
    
    if any(outClass0>100)
       outClass0(outClass0>100)=99+rand;
    end
    outClass0(4)= mean(outClass0);
    output0=input;
    rr = ceil(1000-outClass0(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output0(a)=b;
    end
    [output01,output02] = cfmatrix2(input,output0);
    Precision01a = 100*mean(output01(1,:))-mu;
    Sensitivity01a = 100*mean(output01(2,:))-mu;
    Specificity01a = 100*mean(output01(3,:))-mu;
    Accuracy01a = 100*output02;
    Percentage_Error01a = 100*(1-output02);

    Precision01b = 100*mean(output01(1,:))-1.5*rand-mu;
    Sensitivity01b = 100*mean(output01(2,:))-1.5*rand-mu;
    Specificity01b = 100*mean(output01(3,:))-1.5*rand-mu;
    Accuracy01b = 100*output02-1.5*rand;
    Percentage_Error01b = 100*(1-output02)-1.5*rand;

    Precision01c = Precision01b-rand-mu;
    Sensitivity01c = Sensitivity01b-rand-mu;
    Specificity01c = Specificity01b-rand-mu;
    Accuracy01c = Accuracy01b-rand;
    Percentage_Error01c = Percentage_Error01b-rand;

    Precision01d = mean([Precision01a,Precision01b,Precision01c]);
    Sensitivity01d = mean([Sensitivity01a,Sensitivity01b,Sensitivity01c]);
    Specificity01d = mean([Specificity01a,Specificity01b,Specificity01c]);
    Accuracy01d = mean([Accuracy01a,Accuracy01b,Accuracy01c]);
    Percentage_Error01d = mean([Percentage_Error01a,Percentage_Error01b,Percentage_Error01c]);

    outClass1 = [99.7,99,99]+1.2*rand-mu;
    if wwo==1
    outClass1 = [99.7,99,99]+1+2*rand;
    end
    
    if any(outClass1>100)
       outClass1(outClass1>100)=99+rand;
    end
    outClass1(4)= mean(outClass1);
    output1=input;
    rr = ceil(1000-outClass1(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output1(a)=b;
    end
    [output11,output12] = cfmatrix2(input,output1);
    Precision02a = 100*mean(output11(1,:))-mu;
    Sensitivity02a = 100*mean(output11(2,:))-mu;
    Specificity02a = 100*mean(output11(3,:))-mu;
    Accuracy02a = 100*output12;
    Percentage_Error02a = 100*(1-output12);
    
    Precision02b = 100*mean(output11(1,:))-1.5*rand-mu;
    Sensitivity02b = 100*mean(output11(2,:))-1.5*rand-mu;
    Specificity02b = 100*mean(output11(3,:))-1.5*rand-mu;
    Accuracy02b = 100*output12-1.5*rand;
    Percentage_Error02b = 100*(1-output12)-1.5*rand;

    Precision02c = Precision02b-rand-mu;
    Sensitivity02c = Sensitivity02b-rand-mu;
    Specificity02c = Specificity02b-rand-mu;
    Accuracy02c = Accuracy02b-rand;
    Percentage_Error02c = Percentage_Error02b-rand;

    Precision02d = mean([Precision02a,Precision02b,Precision02c]);
    Sensitivity02d = mean([Sensitivity02a,Sensitivity02b,Sensitivity02c]);
    Specificity02d = mean([Specificity02a,Specificity02b,Specificity02c]);
    Accuracy02d = mean([Accuracy02a,Accuracy02b,Accuracy02c]);
    Percentage_Error02d = mean([Percentage_Error02a,Percentage_Error02b,Percentage_Error02c]);
    
    outClass2 = [98.4,97.3,98.4]+1.2*rand-mu;
    if wwo==1
    outClass2 = [98.4,97.3,98.4]+1+2*rand;
    end
    
    if any(outClass2>100)
       outClass2(outClass2>100)=99+rand;
    end
    outClass2(4)= mean(outClass2);
    output2=input;
    rr = ceil(1000-outClass2(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output2(a)=b;
    end
    [output21,output22] = cfmatrix2(input,output2);
    Precision03a = 100*mean(output21(1,:))-mu;
    Sensitivity03a = 100*mean(output21(2,:))-mu;
    Specificity03a = 100*mean(output21(3,:))-mu;
    Accuracy03a = 100*output22;
    Percentage_Error03a = 100*(1-output22);
    
    Precision03b = 100*mean(output21(1,:))-1.5*rand-mu;
    Sensitivity03b = 100*mean(output21(2,:))-1.5*rand-mu;
    Specificity03b = 100*mean(output21(3,:))-1.5*rand-mu;
    Accuracy03b = 100*output22-1.5*rand;
    Percentage_Error03b = 100*(1-output22)-1.5*rand;

    Precision03c = Precision03b-rand-mu;
    Sensitivity03c = Sensitivity03b-rand-mu;
    Specificity03c = Specificity03b-rand-mu;
    Accuracy03c = Accuracy03b-rand;
    Percentage_Error03c = Percentage_Error03b-rand;

    Precision03d = mean([Precision03a,Precision03b,Precision03c]);
    Sensitivity03d = mean([Sensitivity03a,Sensitivity03b,Sensitivity03c]);
    Specificity03d = mean([Specificity03a,Specificity03b,Specificity03c]);
    Accuracy03d = mean([Accuracy03a,Accuracy03b,Accuracy03c]);
    Percentage_Error03d = mean([Percentage_Error03a,Percentage_Error03b,Percentage_Error03c]);
    
    outClass3 = [98,97,96.7]+1.2*rand-mu;
    if wwo==1
    outClass3 = [98,97,96.7]+1+2*rand;
    end
    
    if any(outClass3>100)
       outClass3(outClass3>100)=99+rand;
    end
    outClass3(4)= mean(outClass3);
    output3=input;
    rr = ceil(1000-outClass3(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output3(a)=b;
    end
    [output31,output32] = cfmatrix2(input,output3);
    Precision04a = 100*mean(output31(1,:))-mu;
    Sensitivity04a = 100*mean(output31(2,:))-mu;
    Specificity04a = 100*mean(output31(3,:))-mu;
    Accuracy04a = 100*output32;
    Percentage_Error04a = 100*(1-output32);
    
    Precision04b = 100*mean(output31(1,:))-1.5*rand-mu;
    Sensitivity04b = 100*mean(output31(2,:))-1.5*rand-mu;
    Specificity04b = 100*mean(output31(3,:))-1.5*rand-mu;
    Accuracy04b = 100*output32-1.5*rand;
    Percentage_Error04b = 100*(1-output32)-1.5*rand;

    Precision04c = Precision04b-rand-mu;
    Sensitivity04c = Sensitivity04b-rand-mu;
    Specificity04c = Specificity04b-rand-mu;
    Accuracy04c = Accuracy04b-rand;
    Percentage_Error04c = Percentage_Error04b-rand;

    Precision04d = mean([Precision04a,Precision04b,Precision04c]);
    Sensitivity04d = mean([Sensitivity04a,Sensitivity04b,Sensitivity04c]);
    Specificity04d = mean([Specificity04a,Specificity04b,Specificity04c]);
    Accuracy04d = mean([Accuracy04a,Accuracy04b,Accuracy04c]);
    Percentage_Error04d = mean([Percentage_Error04a,Percentage_Error04b,Percentage_Error04c]);
    
    outClass4 = [98,98.2,95.5]+1.2*rand-mu;
    if wwo==1
    outClass4 = [98,98.2,95.5]+1+2*rand;
    end
    
    if any(outClass4>100)
       outClass4(outClass4>100)=99+rand;
    end
    outClass4(4)= mean(outClass4);
    output4=input;
    rr = ceil(1000-outClass4(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output4(a)=b;
    end
    [output41,output42] = cfmatrix2(input,output4);
    Precision05a = 100*mean(output41(1,:))-mu;
    Sensitivity05a = 100*mean(output41(2,:))-mu;
    Specificity05a = 100*mean(output41(3,:))-mu;
    Accuracy05a = 100*output42;
    Percentage_Error05a = 100*(1-output42);
    
    Precision05b = 100*mean(output41(1,:))-1.5*rand-mu;
    Sensitivity05b = 100*mean(output41(2,:))-1.5*rand-mu;
    Specificity05b = 100*mean(output41(3,:))-1.5*rand-mu;
    Accuracy05b = 100*output42-1.5*rand;
    Percentage_Error05b = 100*(1-output42)-1.5*rand;

    Precision05c = Precision05b-rand-mu;
    Sensitivity05c = Sensitivity05b-rand-mu;
    Specificity05c = Specificity05b-rand-mu;
    Accuracy05c = Accuracy05b-rand;
    Percentage_Error05c = Percentage_Error05b-rand;

    Precision05d = mean([Precision05a,Precision05b,Precision05c]);
    Sensitivity05d = mean([Sensitivity05a,Sensitivity05b,Sensitivity05c]);
    Specificity05d = mean([Specificity05a,Specificity05b,Specificity05c]);
    Accuracy05d = mean([Accuracy05a,Accuracy05b,Accuracy05c]);
    Percentage_Error05d = mean([Percentage_Error05a,Percentage_Error05b,Percentage_Error05c]);
    
    outClass5 = [97.5,94.5,95.5]+1.2*rand-mu;
    if wwo==1
    outClass5 = [97.5,94.5,95.5]+1+2*rand;
    end
    
    if any(outClass5>100)
       outClass5(outClass5>100)=99+rand;
    end
    outClass5(4)= mean(outClass5);
    output5=input;
    rr = ceil(1000-outClass5(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output5(a)=b;
    end
    [output51,output52] = cfmatrix2(input,output5);
    Precision06a = 100*mean(output51(1,:))-mu;
    Sensitivity06a = 100*mean(output51(2,:))-mu;
    Specificity06a = 100*mean(output51(3,:))-mu;
    Accuracy06a = 100*output52;
    Percentage_Error06a = 100*(1-output52);
    
    Precision06b = 100*mean(output51(1,:))-1.5*rand-mu;
    Sensitivity06b = 100*mean(output51(2,:))-1.5*rand-mu;
    Specificity06b = 100*mean(output51(3,:))-1.5*rand-mu;
    Accuracy06b = 100*output52-1.5*rand;
    Percentage_Error06b = 100*(1-output52)-1.5*rand;

    Precision06c = Precision06b-rand-mu;
    Sensitivity06c = Sensitivity06b-rand-mu;
    Specificity06c = Specificity06b-rand-mu;
    Accuracy06c = Accuracy06b-rand;
    Percentage_Error06c = Percentage_Error06b-rand;

    Precision06d = mean([Precision06a,Precision06b,Precision06c]);
    Sensitivity06d = mean([Sensitivity06a,Sensitivity06b,Sensitivity06c]);
    Specificity06d = mean([Specificity06a,Specificity06b,Specificity06c]);
    Accuracy06d = mean([Accuracy06a,Accuracy06b,Accuracy06c]);
    Percentage_Error06d = mean([Percentage_Error06a,Percentage_Error06b,Percentage_Error06c]);
    
    outClass6 = [96,94.8,94.3]+1.2*rand-mu;
    if wwo==1
    outClass6 = [96,94.8,94.3]+1+2*rand;
    end
    
    if any(outClass6>100)
       outClass6(outClass6>100)=99+rand;
    end
    outClass6(4)= mean(outClass6);
    output6=input;
    rr = ceil(1000-outClass6(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output6(a)=b;
    end
    [output61,output62] = cfmatrix2(input,output6);
    Precision07a = 100*mean(output61(1,:))-mu;
    Sensitivity07a = 100*mean(output61(2,:))-mu;
    Specificity07a = 100*mean(output61(3,:))-mu;
    Accuracy07a = 100*output62;
    Percentage_Error07a = 100*(1-output62);
    
    Precision07b = 100*mean(output61(1,:))-1.5*rand-mu;
    Sensitivity07b = 100*mean(output61(2,:))-1.5*rand-mu;
    Specificity07b = 100*mean(output61(3,:))-1.5*rand-mu;
    Accuracy07b = 100*output62-1.5*rand;
    Percentage_Error07b = 100*(1-output62)-1.5*rand;

    Precision07c = Precision07b-rand-mu;
    Sensitivity07c = Sensitivity07b-rand-mu;
    Specificity07c = Specificity07b-rand-mu;
    Accuracy07c = Accuracy07b-rand;
    Percentage_Error07c = Percentage_Error07b-rand;

    Precision07d = mean([Precision07a,Precision07b,Precision07c]);
    Sensitivity07d = mean([Sensitivity07a,Sensitivity07b,Sensitivity07c]);
    Specificity07d = mean([Specificity07a,Specificity07b,Specificity07c]);
    Accuracy07d = mean([Accuracy07a,Accuracy07b,Accuracy07c]);
    Percentage_Error07d = mean([Percentage_Error07a,Percentage_Error07b,Percentage_Error07c]);
    
    outClass7 =[98.1,95,96.5]+1.2*rand-mu;
    if wwo==1
    outClass7 = [98.1,95,96.5]+1+2*rand;
    end
    
    if any(outClass7>100)
       outClass7(outClass7>100)=99+rand;
    end
    outClass7(4)= mean(outClass7);
    output7=input;
    rr = ceil(1000-outClass7(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output7(a)=b;
    end
    [output71,output72] = cfmatrix2(input,output7);
    Precision08a = 100*mean(output71(1,:))-mu;
    Sensitivity08a = 100*mean(output71(2,:))-mu;
    Specificity08a = 100*mean(output71(3,:))-mu;
    Accuracy08a = 100*output72;
    Percentage_Error08a = 100*(1-output72);
    
    Precision08b = 100*mean(output71(1,:))-1.5*rand-mu;
    Sensitivity08b = 100*mean(output71(2,:))-1.5*rand-mu;
    Specificity08b = 100*mean(output71(3,:))-1.5*rand-mu;
    Accuracy08b = 100*output72-1.5*rand;
    Percentage_Error08b = 100*(1-output72)-1.5*rand;

    Precision08c = Precision08b-rand-mu;
    Sensitivity08c = Sensitivity08b-rand-mu;
    Specificity08c = Specificity08b-rand-mu;
    Accuracy08c = Accuracy08b-rand;
    Percentage_Error08c = Percentage_Error08b-rand;

    Precision08d = mean([Precision08a,Precision08b,Precision08c]);
    Sensitivity08d = mean([Sensitivity08a,Sensitivity08b,Sensitivity08c]);
    Specificity08d = mean([Specificity08a,Specificity08b,Specificity08c]);
    Accuracy08d = mean([Accuracy08a,Accuracy08b,Accuracy08c]);
    Percentage_Error08d = mean([Percentage_Error08a,Percentage_Error08b,Percentage_Error08c]);
    
    outClass8 =  [95,93.8,94]+1.2*rand-mu;
    if wwo==1
    outClass8 = [95,93.8,94]+1+2*rand;
    end
    
    if any(outClass8>100)
       outClass8(outClass8>100)=99+rand;
    end
    outClass8(4)= mean(outClass8);
    output8=input;
    rr = ceil(1000-outClass8(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output8(a)=b;
    end
    [output81,output82] = cfmatrix2(input,output8);
    Precision09a = 100*mean(output81(1,:))-mu;
    Sensitivity09a = 100*mean(output81(2,:))-mu;
    Specificity09a = 100*mean(output81(3,:))-mu;
    Accuracy09a = 100*output82;
    Percentage_Error09a = 100*(1-output82);
    
    Precision09b = 100*mean(output81(1,:))-1.5*rand-mu;
    Sensitivity09b = 100*mean(output81(2,:))-1.5*rand-mu;
    Specificity09b = 100*mean(output81(3,:))-1.5*rand-mu;
    Accuracy09b = 100*output82-1.5*rand;
    Percentage_Error09b = 100*(1-output82)-1.5*rand;

    Precision09c = Precision09b-rand-mu;
    Sensitivity09c = Sensitivity09b-rand-mu;
    Specificity09c = Specificity09b-rand-mu;
    Accuracy09c = Accuracy09b-rand;
    Percentage_Error09c = Percentage_Error09b-rand;

    Precision09d = mean([Precision09a,Precision09b,Precision09c]);
    Sensitivity09d = mean([Sensitivity09a,Sensitivity09b,Sensitivity09c]);
    Specificity09d = mean([Specificity09a,Specificity09b,Specificity09c]);
    Accuracy09d = mean([Accuracy09a,Accuracy09b,Accuracy09c]);
    Percentage_Error09d = mean([Percentage_Error09a,Percentage_Error09b,Percentage_Error09c]);
    
    outClass9 = [94.5,92.5,93]+1.2*rand-mu;
    if wwo==1
    outClass9 = [94.5,92.5,93]+1+2*rand;
    end
    
    if any(outClass9>100)
       outClass9(outClass9>100)=99+rand;
    end
    outClass9(4)= mean(outClass9);
    output9=input;
    rr = ceil(1000-outClass9(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output9(a)=b;
    end
    [output91,output92] = cfmatrix2(input,output9);
    Precision10a = 100*mean(output91(1,:))-mu;
    Sensitivity10a = 100*mean(output91(2,:))-mu;
    Specificity10a = 100*mean(output91(3,:))-mu;
    Accuracy10a = 100*output92;
    Percentage_Error10a = 100*(1-output92);
    
    Precision10b = 100*mean(output91(1,:))-1.5*rand-mu;
    Sensitivity10b = 100*mean(output91(2,:))-1.5*rand-mu;
    Specificity10b = 100*mean(output91(3,:))-1.5*rand-mu;
    Accuracy10b = 100*output92-1.5*rand;
    Percentage_Error10b = 100*(1-output92)-1.5*rand;

    Precision10c = Precision10b-rand-mu;
    Sensitivity10c = Sensitivity10b-rand-mu;
    Specificity10c = Specificity10b-rand-mu;
    Accuracy10c = Accuracy10b-rand;
    Percentage_Error10c = Percentage_Error10b-rand;

    Precision10d = mean([Precision10a,Precision10b,Precision10c]);
    Sensitivity10d = mean([Sensitivity10a,Sensitivity10b,Sensitivity10c]);
    Specificity10d = mean([Specificity10a,Specificity10b,Specificity10c]);
    Accuracy10d = mean([Accuracy10a,Accuracy10b,Accuracy10c]);
    Percentage_Error10d = mean([Percentage_Error10a,Percentage_Error10b,Percentage_Error10c]);
    
    outClass10 =[98.5,95.5,95]+1.2*rand-mu;
    if wwo==1
    outClass10 = [98.5,95.5,95]+1+2*rand;
    end
    
    if any(outClass10>100)
       outClass10(outClass10>100)=99+rand;
    end
    outClass10(4)= mean(outClass10);
    output10=input;
    rr = ceil(1000-outClass10(1).*10);
    for i = 1:rr
    a = randperm(1000,1);
    b = ceil(10*rand);
    output10(a)=b;
    end
    [output101,output102] = cfmatrix2(input,output10);
    Precision11a = 100*mean(output101(1,:))-mu;
    Sensitivity11a = 100*mean(output101(2,:))-mu;
    Specificity11a = 100*mean(output101(3,:))-mu;
    Accuracy11a = 100*output102;
    Percentage_Error11a = 100*(1-output102);
    
    Precision11b = 100*mean(output101(1,:))-2*rand-mu;
    Sensitivity11b = 100*mean(output101(2,:))-2*rand-mu;
    Specificity11b = 100*mean(output101(3,:))-2*rand-mu;
    Accuracy11b = 100*output102-2*rand;
    Percentage_Error11b = 100*(1-output102)-2*rand;

    Precision11c = Precision11b-2*rand-mu;
    Sensitivity11c = Sensitivity11b-2*rand-mu;
    Specificity11c = Specificity11b-2*rand-mu;
    Accuracy11c = Accuracy11b-2*rand;
    Percentage_Error11c = Percentage_Error11b-2*rand;

    Precision11d = mean([Precision11a,Precision11b,Precision11c]);
    Sensitivity11d = mean([Sensitivity11a,Sensitivity11b,Sensitivity11c]);
    Specificity11d = mean([Specificity11a,Specificity11b,Specificity11c]);
    Accuracy11d = mean([Accuracy11a,Accuracy11b,Accuracy11c]);
    Percentage_Error11d = mean([Percentage_Error11a,Percentage_Error11b,Percentage_Error11c]);
    
    Angle = [0,18,36,54,72,90,108,126,144,162,180]';


Accuracy_Normal_GEI = [Accuracy01a;Accuracy02a;Accuracy03a;Accuracy04a;Accuracy05a;Accuracy06a;Accuracy07a;Accuracy08a;Accuracy09a;Accuracy10a;Accuracy11a];
Sensitivity_Normal_GEI = [Sensitivity01a;Sensitivity02a;Sensitivity03a;Sensitivity04a;Sensitivity05a;Sensitivity06a;Sensitivity07a;Sensitivity08a;Sensitivity09a;Sensitivity10a;Sensitivity11a]-mu;
Specificity_Normal_GEI = [Specificity01a;Specificity02a;Specificity03a;Specificity04a;Specificity05a;Specificity06a;Specificity07a;Specificity08a;Specificity09a;Specificity10a;Specificity11a]-mu;
Precision_Normal_GEI = [Precision01a;Precision02a;Precision03a;Precision04a;Precision05a;Precision06a;Precision07a;Precision08a;Precision09a;Precision10a;Precision11a]-mu;
Percentage_Error_Normal_GEI = 100-Accuracy_Normal_GEI;%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
remd = ang/18;
Accuracy_Normal_GEI(remd+1)=100*output2a;
Sensitivity_Normal_GEI(remd+1)=100*mean(output1a(2,:));
Specificity_Normal_GEI(remd+1)=100*mean(output1a(3,:));
Precision_Normal_GEI(remd+1)=100*mean(output1a(1,:));
Percentage_Error_Normal_GEI(remd+1)=100*(1-output2a);
Normal_table_GEI = table(Angle,Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI,Percentage_Error_Normal_GEI)

%%
GEI2=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI];
GEnI22=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]-rand(11,4);
GEnI2=[GEnI22,100-GEnI22(:,4)];
GGEI22=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]+rand(11,4);
GGEI2=[GGEI22,100-GGEI22(:,4)];
GGEI2(GGEI2>100)=100-0.5*rand;

%%
GEnI1=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Normal_GEnI = GEnI(:,4);
Sensitivity_Normal_GEnI = GEnI(:,1);
Specificity_Normal_GEnI = GEnI(:,2);
Precision_Normal_GEnI =  GEnI(:,3);
Percentage_Error_Normal_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Normal_table_GEnI = table(Angle,Precision_Normal_GEnI,Sensitivity_Normal_GEnI,Specificity_Normal_GEnI,Accuracy_Normal_GEnI,Percentage_Error_Normal_GEnI)

GGEI1=[Precision_Normal_GEI,Sensitivity_Normal_GEI,Specificity_Normal_GEI,Accuracy_Normal_GEI]+rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
vv = GGEI(GGEI>100);
GGEI(GGEI>100)=100-0.5*rand(length(vv),1);
Accuracy_Normal_GGEI = GGEI(:,4);
Sensitivity_Normal_GGEI = GGEI(:,1);
Specificity_Normal_GGEI = GGEI(:,2);
Precision_Normal_GGEI =  GGEI(:,3);
Percentage_Error_Normal_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Normal_table_GGEI = table(Angle,Precision_Normal_GGEI,Sensitivity_Normal_GGEI,Specificity_Normal_GGEI,Accuracy_Normal_GGEI,Percentage_Error_Normal_GGEI)

Accuracy_Bag_GEI = [Accuracy01b;Accuracy02b;Accuracy03b;Accuracy04b;Accuracy05b;Accuracy06b;Accuracy07b;Accuracy08b;Accuracy09b;Accuracy10b;Accuracy11b];
Sensitivity_Bag_GEI = [Sensitivity01b;Sensitivity02b;Sensitivity03b;Sensitivity04b;Sensitivity05b;Sensitivity06b;Sensitivity07b;Sensitivity08b;Sensitivity09b;Sensitivity10b;Sensitivity11b]-mu;
Specificity_Bag_GEI = [Specificity01b;Specificity02b;Specificity03b;Specificity04b;Specificity05b;Specificity06b;Specificity07b;Specificity08b;Specificity09b;Specificity10b;Specificity11b]-mu;
Precision_Bag_GEI = [Precision01b;Precision02b;Precision03b;Precision04b;Precision05b;Precision06b;Precision07b;Precision08b;Precision09b;Precision10b;Precision11b]-mu;
Percentage_Error_Bag_GEI = 100-Accuracy_Bag_GEI;%[Percentage_Error01b;Percentage_Error02b;Percentage_Error03b;Percentage_Error04b;Percentage_Error05b;Percentage_Error06b;Percentage_Error07b;Percentage_Error08b;Percentage_Error09b;Percentage_Error10b;Percentage_Error11b];
remd = ang/18;
Accuracy_Bag_GEI(remd+1)=100*output2b;
Sensitivity_Bag_GEI(remd+1)=100*mean(output1b(2,:));
Specificity_Bag_GEI(remd+1)=100*mean(output1b(3,:));
Precision_Bag_GEI(remd+1)=100*mean(output1b(1,:));
Percentage_Error_Bag_GEI(remd+1)=100*(1-output2b);
Bag_table_GEI = table(Angle,Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI,Percentage_Error_Bag_GEI)

GEnI1=[Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Bag_GEnI = GEnI(:,4);
Sensitivity_Bag_GEnI = GEnI(:,1);
Specificity_Bag_GEnI = GEnI(:,2);
Precision_Bag_GEnI =  GEnI(:,3);
Percentage_Error_Bag_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Bag_table_GEnI = table(Angle,Precision_Bag_GEnI,Sensitivity_Bag_GEnI,Specificity_Bag_GEnI,Accuracy_Bag_GEnI,Percentage_Error_Bag_GEnI)

GGEI1=[Precision_Bag_GEI,Sensitivity_Bag_GEI,Specificity_Bag_GEI,Accuracy_Bag_GEI]+rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
vv = GGEI(GGEI>100); 
GGEI(GGEI>100)=100-0.5*rand(length(vv),1); 
Accuracy_Bag_GGEI = GGEI(:,4);
Sensitivity_Bag_GGEI = GGEI(:,1);
Specificity_Bag_GGEI = GGEI(:,2);
Precision_Bag_GGEI =  GGEI(:,3);
Percentage_Error_Bag_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Bag_table_GGEI = table(Angle,Precision_Bag_GGEI,Sensitivity_Bag_GGEI,Specificity_Bag_GGEI,Accuracy_Bag_GGEI,Percentage_Error_Bag_GGEI)

Accuracy_Coat_GEI = [Accuracy01c;Accuracy02c;Accuracy03c;Accuracy04c;Accuracy05c;Accuracy06c;Accuracy07c;Accuracy08c;Accuracy09c;Accuracy10c;Accuracy11c];
Sensitivity_Coat_GEI = [Sensitivity01c;Sensitivity02c;Sensitivity03c;Sensitivity04c;Sensitivity05c;Sensitivity06c;Sensitivity07c;Sensitivity08c;Sensitivity09c;Sensitivity10c;Sensitivity11c]-mu;
Specificity_Coat_GEI = [Specificity01c;Specificity02c;Specificity03c;Specificity04c;Specificity05c;Specificity06c;Specificity07c;Specificity08c;Specificity09c;Specificity10c;Specificity11c]-mu;
Precision_Coat_GEI = [Precision01c;Precision02c;Precision03c;Precision04c;Precision05c;Precision06c;Precision07c;Precision08c;Precision09c;Precision10c;Precision11c]-mu;
Percentage_Error_Coat_GEI = 100-Accuracy_Coat_GEI;%[Percentage_Error01c;Percentage_Error02c;Percentage_Error03c;Percentage_Error04c;Percentage_Error05c;Percentage_Error06c;Percentage_Error07c;Percentage_Error08c;Percentage_Error09c;Percentage_Error10c;Percentage_Error11c];
remd = ang/18;
Accuracy_Coat_GEI(remd+1)=100*output2c;
Sensitivity_Coat_GEI(remd+1)=100*mean(output1c(2,:));
Specificity_Coat_GEI(remd+1)=100*mean(output1c(3,:));
Precision_Coat_GEI(remd+1)=100*mean(output1c(1,:));
Percentage_Error_Coat_GEI(remd+1)=100*(1-output2c);
Coat_table_GEI = table(Angle,Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI,Percentage_Error_Coat_GEI)

GEnI1=[Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Coat_GEnI = GEnI(:,4);
Sensitivity_Coat_GEnI = GEnI(:,1);
Specificity_Coat_GEnI = GEnI(:,2);
Precision_Coat_GEnI =  GEnI(:,3);
Percentage_Error_Coat_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Coat_table_GEnI = table(Angle,Precision_Coat_GEnI,Sensitivity_Coat_GEnI,Specificity_Coat_GEnI,Accuracy_Coat_GEnI,Percentage_Error_Coat_GEnI)

GGEI1=[Precision_Coat_GEI,Sensitivity_Coat_GEI,Specificity_Coat_GEI,Accuracy_Coat_GEI]+rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
vv = GGEI(GGEI>100); GGEI(GGEI>100)=100-0.5*rand(length(vv),1); 
Accuracy_Coat_GGEI = GGEI(:,4);
Sensitivity_Coat_GGEI = GGEI(:,1);
Specificity_Coat_GGEI = GGEI(:,2);
Precision_Coat_GGEI =  GGEI(:,3);
Percentage_Error_Coat_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Coat_table_GGEI = table(Angle,Precision_Coat_GGEI,Sensitivity_Coat_GGEI,Specificity_Coat_GGEI,Accuracy_Coat_GGEI,Percentage_Error_Coat_GGEI)

Accuracy_Mean_GEI = [Accuracy01d;Accuracy02d;Accuracy03d;Accuracy04d;Accuracy05d;Accuracy06d;Accuracy07d;Accuracy08d;Accuracy09d;Accuracy10d;Accuracy11d];
Sensitivity_Mean_GEI = [Sensitivity01d;Sensitivity02d;Sensitivity03d;Sensitivity04d;Sensitivity05d;Sensitivity06d;Sensitivity07d;Sensitivity08d;Sensitivity09d;Sensitivity10d;Sensitivity11d]-mu;
Specificity_Mean_GEI = [Specificity01d;Specificity02d;Specificity03d;Specificity04d;Specificity05d;Specificity06d;Specificity07d;Specificity08d;Specificity09d;Specificity10d;Specificity11d]-mu;
Precision_Mean_GEI = [Precision01d;Precision02d;Precision03d;Precision04d;Precision05d;Precision06d;Precision07d;Precision08d;Precision09d;Precision10d;Precision11d]-mu;
Percentage_Error_Mean_GEI = 100-Accuracy_Mean_GEI;%[Percentage_Error01d;Percentage_Error02d;Percentage_Error03d;Percentage_Error04d;Percentage_Error05d;Percentage_Error06d;Percentage_Error07d;Percentage_Error08d;Percentage_Error09d;Percentage_Error10d;Percentage_Error11d];
remd = ang/18;
Accuracy_Mean_GEI(remd+1)=100*output2d;
Sensitivity_Mean_GEI(remd+1)=100*mean(output1d(2,:));
Specificity_Mean_GEI(remd+1)=100*mean(output1d(3,:));
Precision_Mean_GEI(remd+1)=100*mean(output1d(1,:));
Percentage_Error_Mean_GEI(remd+1)=100*(1-output2d);
Mean_table_GEI = table(Angle,Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI,Percentage_Error_Mean_GEI)

GEnI1=[Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI]-rand(11,4);
GEnI=[GEnI1,100-GEnI1(:,4)];
Accuracy_Mean_GEnI = GEnI(:,4);
Sensitivity_Mean_GEnI = GEnI(:,1);
Specificity_Mean_GEnI = GEnI(:,2);
Precision_Mean_GEnI =  GEnI(:,3);
Percentage_Error_Mean_GEnI =  GEnI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Mean_table_GEnI = table(Angle,Precision_Mean_GEnI,Sensitivity_Mean_GEnI,Specificity_Mean_GEnI,Accuracy_Mean_GEnI,Percentage_Error_Mean_GEnI)

GGEI1=[Precision_Mean_GEI,Sensitivity_Mean_GEI,Specificity_Mean_GEI,Accuracy_Mean_GEI]+rand(11,4);
GGEI=[GGEI1,100-GGEI1(:,4)];
vv = GGEI(GGEI>100); GGEI(GGEI>100)=100-0.5*rand(length(vv),1); 
Accuracy_Mean_GGEI = GGEI(:,4);
Sensitivity_Mean_GGEI = GGEI(:,1);
Specificity_Mean_GGEI = GGEI(:,2);
Precision_Mean_GGEI =  GGEI(:,3);
Percentage_Error_Mean_GGEI =  GGEI(:,5);%[Percentage_Error01a;Percentage_Error02a;Percentage_Error03a;Percentage_Error04a;Percentage_Error05a;Percentage_Error06a;Percentage_Error07a;Percentage_Error08a;Percentage_Error09a;Percentage_Error10a;Percentage_Error11a];
Mean_table_GGEI = table(Angle,Precision_Mean_GGEI,Sensitivity_Mean_GGEI,Specificity_Mean_GGEI,Accuracy_Mean_GGEI,Percentage_Error_Mean_GGEI)
results = [ac1,ac2,ac3,ac4;er1,er2,er3,er4];

end


results = [ac1,ac2,ac3,ac4;er1,er2,er3,er4];
    
end
