tic
clc;
clear;
close all;
warning off

global MaxIt nPop mu method
flag=1;
ang = 90;%input('Enter the angle : ');
list = dir('D:\Ph.D\Ph.D. WORK codes\OU_MVLP_Dataset\**');
kk=ang;
kkk=num2str(kk);
          if kk < 10
              kk1=strcat('00',kkk);
          elseif  kk<100
              kk1=strcat('0',kkk);
          else
              kk1 = kkk;
          end
ii=1;
for i = 1:size(list,1)
    if length(list(i).name)>15
    ang1 = list(i).name(end-6:end-4);

    if strcmp(ang1,kk1) && strcmp(list(i).name(end),'g') 
    img{ii}=imresize(double(imread(fullfile(list(i).folder,list(i).name))),[240,240]);
    %figure(1),imshow(img1)
    ii=ii+1;
    end
    end
end

for i=1:length(img)
s = img{1,i};
% figure,imshow(s)
% title('Gait Energy Image')
grad{i} = imgradient(s); % takes 3x3 gradient of 
% figure,imshow(grad{i},[])
% title('Gradient Image')
end
r1=1;
r2=240;
%% Problem Definition
rows = [r1,r2];
CostFunction=@(x,img,grad,ang) Sphere1(x,img,grad,ang);     % Cost Function

nVar=28;                 % Number of Decision Variables

VarSize= 1;       % Decision Variables Matrix Size
VarMin1= r1+35;         % Lower Bound of Variables
VarMax1= r1+50;         % Upper Bound of Variables

VarMin2= r2-50;         % Lower Bound of Variables
VarMax2= r2-25;         % Upper Bound of Variables

VarMin3= 80;          % Lower Bound of Variables
VarMax3= 160;         % Upper Bound of Variables
method =2;
global MaxIt nPop mu 
if flag==0
%% Firefly Algorithm Parameters
MaxIt=25;         % Maximum Number of Iterations

nPop=25;            % Number of Fireflies (Swarm Size)

gamma=1;            % Light Absorption Coefficient

beta0=2;            % Attraction Coefficient Base Value

alpha=0.2;          % Mutation Coefficient

alpha_damp=0.98;    % Mutation Coefficient Damping Ratio

delta1=0.05*(VarMax1-VarMin1);     % Uniform Mutation Range
delta2=0.05*(VarMax2-VarMin2);     % Uniform Mutation Range
delta3=0.05*(VarMax3-VarMin3);     % Uniform Mutation Range

pl = 1+rand(2-1); % power law index for levy flights

mu=.01;

if isscalar(VarMin1) && isscalar(VarMax1)
    dmax1 = (VarMax1-VarMin1)*sqrt(nVar);
    dmax2 = (VarMax2-VarMin2)*sqrt(nVar);
    dmax3 = (VarMax3-VarMin3)*sqrt(nVar);
else
    dmax1 = norm(VarMax1-VarMin1);
    dmax2 = norm(VarMax2-VarMin3);
    dmax3 = norm(VarMax2-VarMin3);
end

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=0;

% Create Initial Fireflies
for i=1:nPop
    Position1=round(unifrnd(VarMin1,VarMax1,VarSize));
    Position2=round(unifrnd(VarMin2,VarMax2,VarSize));
    Position3=round(unifrnd(VarMin3,VarMax3,VarSize));
    Position4=1;
    Position5=0;
    Position6=0;
    Position7=1;
    pop(i).Position = [Position1,Position2,Position3,Position4,Position5,Position6,Position7];
    pop(i).Cost=CostFunction(pop(i).Position,img,grad,ang);
   
   if pop(i).Cost>=BestSol.Cost
       BestSol=pop(i);
   end
end

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop

for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Cost = 0;
        for j=1:nPop
            if pop(j).Cost > pop(i).Cost
                for ii=1:nPop 
                if isempty(pop(i).Position)
                pop(i).Position=(pop(ii).Position);
                end
                if isempty(pop(j).Position)
                pop(j).Position=(pop(ii).Position);
                end
                end
                gamma_max=2+2*rand;
                gamma_min=.5+.5*rand;
                rij=norm(pop(i).Position-pop(j).Position)/dmax1;
                gamma1 = gamma_max-(gamma_max-gamma_min)*(it/MaxIt)^2;
                beta=beta0*exp(-gamma1*rij^mu);
                beta1=beta0*exp(-gamma1*norm(BestSol.Position-pop(j).Position)/dmax1);
                e=delta1*unifrnd(-1,+1,VarSize);
                %e=delta*randn(VarSize);
                alpha1 = .4/(1+exp(.006*(it-MaxIt))); %modification
                newsol.Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                                + beta1*rand(VarSize).*abs(pop(j).Position-BestSol.Position) ...
                                + alpha1*e*levy(1,1,pl);
                    
                newsol.Position=max(newsol.Position,VarMin1);
                newsol.Position=min(newsol.Position,VarMax1);
                newsol.Position=max(newsol.Position,VarMin2);
                newsol.Position=min(newsol.Position,VarMax2);
                newsol.Position=max(newsol.Position,VarMin3);
                newsol.Position=min(newsol.Position,VarMax3);
                
                newsol.Cost=CostFunction(newsol.Position,img,grad,ang);
                
                if newsol.Cost >= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost>=BestSol.Cost
                        BestSol=newpop(i);
                    end
                end
                
            end
        end
    end
    
    % Merge
    pop=[pop
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    
end

%% Results
results = Sphere2(BestSol.Position,img,grad,flag);
c = categorical({'Accuracy','Error'});
figure,
bar(c,results)
legend('Normal','Bag','Coat','Mean')
title('GEI-FSTS Accuracy & Error Percentage')

elseif flag==1
    %% Firefly Algorithm Parameters
MaxIt=3;         % Maximum Number of Iterations

nPop=1;            % Number of Fireflies (Swarm Size)

gamma=1;            % Light Absorption Coefficient

beta0=2;            % Attraction Coefficient Base Value

alpha=0.2;          % Mutation Coefficient

alpha_damp=0.98;    % Mutation Coefficient Damping Ratio

delta1=0.05*(VarMax1-VarMin1);     % Uniform Mutation Range
delta2=0.05*(VarMax2-VarMin2);     % Uniform Mutation Range
delta3=0.05*(VarMax3-VarMin3);     % Uniform Mutation Range

pl = 1+rand(2-1); % power law index for levy flights

mu=0.01;

if isscalar(VarMin1) && isscalar(VarMax1)
    dmax1 = (VarMax1-VarMin1)*sqrt(nVar);
    dmax2 = (VarMax2-VarMin2)*sqrt(nVar);
    dmax3 = (VarMax3-VarMin3)*sqrt(nVar);
else
    dmax1 = norm(VarMax1-VarMin1);
    dmax2 = norm(VarMax2-VarMin3);
    dmax3 = norm(VarMax2-VarMin3);
end

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=0;

% Create Initial Fireflies
for i=1:nPop
    Position1=round(unifrnd(VarMin1,VarMax1,VarSize));
    Position2=round(unifrnd(VarMin2,VarMax2,VarSize));
    Position3=round(unifrnd(VarMin3,VarMax3,VarSize));
    Position4=1;
    Position5=0;
    Position6=0;
    Position7=1;
    pop(i).Position = [Position1,Position2,Position3,Position4,Position5,Position6,Position7];
    pop(i).Cost=CostFunction(pop(i).Position,img,grad,ang);
   
   if pop(i).Cost>=BestSol.Cost
       BestSol=pop(i);
   end
end

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop

for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Cost = 0;
        for j=1:nPop
            if pop(j).Cost > pop(i).Cost
                for ii=1:nPop 
                if isempty(pop(i).Position)
                pop(i).Position=(pop(ii).Position);
                end
                if isempty(pop(j).Position)
                pop(j).Position=(pop(ii).Position);
                end
                end
            gamma_max=2+2*rand;
                gamma_min=.5+.5*rand;
                rij=norm(pop(i).Position-pop(j).Position)/dmax1;
                gamma1 = gamma_max-(gamma_max-gamma_min)*(it/MaxIt)^2;
                beta=beta0*exp(-gamma1*rij^mu);
                beta1=beta0*exp(-gamma1*norm(BestSol.Position-pop(j).Position)/dmax1);
                e=delta1*unifrnd(-1,+1,VarSize);
                %e=delta*randn(VarSize);
                alpha1 = .4/(1+exp(.006*(it-MaxIt))); %modification
                newsol.Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                                + beta1*rand(VarSize).*abs(pop(j).Position-BestSol.Position) ...
                                + alpha1*e*levy(1,1,pl);
                    
                newsol.Position=max(newsol.Position,VarMin1);
                newsol.Position=min(newsol.Position,VarMax1);
                newsol.Position=max(newsol.Position,VarMin2);
                newsol.Position=min(newsol.Position,VarMax2);
                newsol.Position=max(newsol.Position,VarMin3);
                newsol.Position=min(newsol.Position,VarMax3);
                
                newsol.Cost=CostFunction(newsol.Position,img,grad,ang);
                
                if newsol.Cost >= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost>=BestSol.Cost
                        BestSol=newpop(i);
                    end
                end
                
            end
        end
    end
    
    % Merge
    pop=[pop
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    
end

%% Results
results = Sphere2(BestSol.Position,img,grad,flag);
c = categorical({'Accuracy','Error'});
figure,
bar(c,results)
legend('Normal','Bag','Coat','Mean')
title('GGEI-AFOA Accuracy & Error Percentage')

end
disp('Gait Recognition using Adaptive Firefly Algorithm Completed')
toc