function [output1,output2,confmatrix] = cfmatrix2(actual, predict, classlist, per, printout)
% CFMATRIX2 calculates the confusion matrix for any prediction 
% algorithm ( prediction algorithm generates a list of classes to which 
% each test feature vector is assigned ); 
%
% Outputs: confusion matrix
%
%                 Actual Classes
%                   p       n
%              ___|_____|______| 
%    Predicted  p'|     |      |
%      Classes  n'|     |      |
%
%           Also the TP, FP, FN and TN are output for each class based on 
%           http://en.wikipedia.org/wiki/Confusion_matrix
%           The Precision, Sensitivity and Specificity for each class have
%           also been added in this update along with the overall accuracy
%           of the model ( ModelAccuracy ).
%
%
% Further description of the outputs:
%
% True Postive [TP] = Condition Present + Positive result 
% False Positive [FP] = Condition absent + Positive result [Type I error] 
% False (invalid) Negative [FN] = Condition present + Negative result [Type II error] 
% True (accurate) Negative [TN] = Condition absent + Negative result
% Precision(class) = TP(class) / ( TP(class) + FP(class) )
% Sensitivity(class) = Recall(class) = TruePositiveRate(class)
% = TP(class) / ( TP(class) + FN(class) )
% Specificity ( mostly used in 2 class problems )=
% TrueNegativeRate(class)
% = TN(class) / ( TN(class) + FP(class) )
%
% Inputs: 
% 
% 1. actual / 2. predict
% The inputs provided are the 'actual' classes vector
% and the 'predict'ed classes vector. The actual classes are the classes
% to which the input feature vectors belong. The predicted classes are the 
% class to which the input feature vectors are predicted to belong to, 
% based on a prediction algorithm. 
% The length of actual class vector and the predicted class vector need to 
% be the same. If they are not the same, an error message is displayed. 
% 3. classlist
% The third input provides the list of all the classes {p,n,...} for which 
% the classification is being done. All classes are numbers.
% 4. per = 1/0 (default = 0)
% This parameter when set to 1 provides the values in the confusion matrix 
% as percentages. The default provides the values in numbers.
% 5. printout = 1/0 ( default = 1 )
% This parameter when set to 1 provides output on the matlab terminal and
% can be used to suppress output by setting to 0. ( default = 1 ). Assuming
% 'printout' of output use case would be more common and at the same time 
% provided option to suppress output when the number of classes can be very
% large.
%
% Example:
% >> a = [ 1 2 3 1 2 3 1 1 2 3 2 1 1 2 3];
% >> b = [ 1 2 3 1 2 3 1 1 1 2 2 1 2 1 3];
% >> Cf = cfmatrix2(a, b, [1 2 3], 0, 1); 
% is equivalent to
% >> Cf = cfmatrix2(a, b);
% The values of classlist(unique from actual), per(0), printout(1) are set
% to the respective defaults.
% 
%
% [Avinash Uppuluri: avinash_uv@yahoo.com: Last modified: 03/28/2012]
%
% Changes added for 03/28/2012 upload
% a. Pre-initialize confmatrix
% b. Simplified logic making the code more readable and faster; 
%    (based on comments from an interviewer who reviewed the code)
% c. Provide input variable 'printout' as an option to suppress output to
%    screen ( output to display is still the default (printout = 1) 
%    assuming that will be the more common use case ).
% d. Added Precision(class), Sensitivity(class), Specificity(class) and 
%    the overall accuracy of model calculations.

% If classlist not entered: make classlist equal to all 
% unique elements of actual
if (nargin < 2)
   error('Not enough input arguments. Need atleast two vectors as input');
elseif (nargin == 2)
    classlist = unique(actual); % default values from actual
    per = 0; 
    printout = 1;
elseif (nargin == 3)
    per = 0; % default is numbers and input 1 or higher for percentage
    printout = 1;
elseif (nargin == 4)
    printout = 1; % default is silent output ( 0 ); one or higher printsout
elseif (nargin > 5)
   error('Too many input arguments.');    
end


if (length(actual) ~= length(predict))
    error('First two inputs need to be vectors with equal size.');
elseif ((size(actual,1) ~= 1) && (size(actual,2) ~= 1))
    error('First input needs to be a vector and not a matrix');
elseif ((size(predict,1) ~= 1) && (size(predict,2) ~= 1))
    error('Second input needs to be a vector and not a matrix');
end
format short g;
n_class = length(classlist);
confmatrix = zeros(n_class);
line_two = '----------';
line_three = '_________|';

for i = 1:n_class
    for j = 1:n_class
        m = (predict == classlist(i) ...
           & actual  == classlist(j));
        confmatrix(i,j) = sum(m);
    end
    line_two = strcat(line_two,'---',num2str(classlist(i)),'-----');
    line_three = strcat(line_three,'__________');
end

% True Postive [TP] = Condition Present + Positive result 
% False Positive [FP] = Condition absent + Positive result [Type I error] 
% False (invalid) Negative [FN] = Condition present + Negative result [Type II error] 
% True (accurate) Negative [TN] = Condition absent + Negative result
% Precision(class) = TP(class) / ( TP(class) + FP(class) )
% Sensitivity(class) = Recall(class) = TruePositiveRate(class)
% = TP(class) / ( TP(class) + FN(class) )
% Specificity ( mostly used in 2 class problems )=
% TrueNegativeRate(class)
% = TN(class) / ( TN(class) + FP(class) )
    
TPFPFNTN    = zeros(4, n_class);
Precision   = zeros(1, n_class);
Sensitivity = zeros(1, n_class);
Specificity = zeros(1, n_class);
    
% temps1 = sprintf('    TP  ');
% temps2 = sprintf('    FP  ');
% temps3 = sprintf('    FN  ');
% temps4 = sprintf('    TN  ');
% temps5 = sprintf('Preci.  ');
% temps6 = sprintf('Sensi.  ');
% temps7 = sprintf('Speci.  ');

for i = 1:n_class 
    TPFPFNTN(1, i) = confmatrix(i,i); % TP
%    temps1 = strcat(temps1,sprintf(' |   %2.2f    ',TPFPFNTN(1, i)));
    TPFPFNTN(2, i) = sum(confmatrix(i,:))-confmatrix(i,i); % FP
 %   temps2 = strcat(temps2,sprintf(' |   %2.2f    ',TPFPFNTN(2, i) )); 
    TPFPFNTN(3, i) = sum(confmatrix(:,i))-confmatrix(i,i); % FN
  %  temps3 = strcat(temps3,sprintf(' |   %2.2f    ',TPFPFNTN(3, i) ));  
    TPFPFNTN(4, i) = sum(confmatrix(:)) - sum(confmatrix(i,:)) -...
        sum(confmatrix(:,i)) + confmatrix(i,i); % TN
   % temps4 = strcat(temps4,sprintf(' |   %2.2f    ',TPFPFNTN(4, i) )); 
    % Precision(class) = TP(class) / ( TP(class) + FP(class) )
    Precision(i)   = TPFPFNTN(1, i) / sum(confmatrix(i,:));
    %temps5 = strcat(temps5,sprintf(' |   %1.2f    ',Precision(i) ));
    % Sensitivity(class) = Recall(class) = TruePositiveRate(class)
    % = TP(class) / ( TP(class) + FN(class) )
    Sensitivity(i) = TPFPFNTN(1, i) / sum(confmatrix(:,i));
    %temps6 = strcat(temps6,sprintf(' |   %1.2f    ',Sensitivity(i) ));
    % Specificity ( mostly used in 2 class problems )=
    % TrueNegativeRate(class)
    % = TN(class) / ( TN(class) + FP(class) )
    Specificity(i) = TPFPFNTN(4, i) / ( TPFPFNTN(4, i) + TPFPFNTN(2, i) );
    %temps7 = strcat(temps7,sprintf(' |   %1.2f    ',Specificity(i) ));
end 

ModelAccuracy = sum(diag(confmatrix))/sum(confmatrix(:));
% temps8 = sprintf('Model Accuracy is %1.2f ',ModelAccuracy); 

if (per > 0) % ( if > 0 implies true; < 0 implies false )
    confmatrix = (confmatrix ./ length(actual)).*100;
end

if ( printout > 0 ) % ( if > 0 printout; < 0 no printout )
%     disp('------------------------------------------');
%     disp('             Actual Classes');
%     disp(line_two);
%     disp('Predicted|                     ');
%     disp('  Classes|                     ');
%     disp(line_three);
    
    for i = 1:n_class
  %      temps = sprintf('       %d             ',i);
        for j = 1:n_class
   %         temps = strcat(temps,sprintf(' |    %2.1f    ',confmatrix(i,j)));
        end
%         disp(temps);
        clear temps
    end
%     disp('------------------------------------------');

%     disp('------------------------------------------');
%     disp('             Actual Classes');
%     disp(line_two);
%     disp(temps1); disp(temps2); disp(temps3); disp(temps4);
%     disp(temps5); disp(temps6); disp(temps7);
%     disp('------------------------------------------');
%     disp(temps8);
%     disp('------------------------------------------');
end
clear temps1 temps2 temps3 temps4 temps5 temps6 temps7 temps8
output1 = [Precision;Sensitivity;Specificity];
output2 = ModelAccuracy;