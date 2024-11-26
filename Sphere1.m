
function [z,val]=Sphere1(x,sum,grad,ang)
global method
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
%     figure,imshow(ns{1,k},[])
%     figure,imshow(ng{1,k},[])
%   clear grad1
   load('parameters','grad1')
   pca_s1 = pca(ns{1,k}); 
   pca_grad1 = pca(grad1); 
   en = entropyfilt(uint8(ns{1,k}),ones(9));
   pca_s1 = pca_s1(:)';
   pca_grad1 =pca_grad1(:)';
   pca_en1 = en(:)';
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
z=accuracy;
val=acc(1);
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

end
