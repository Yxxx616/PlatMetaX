classdef UF2 < PROBLEM
% <2009> <multi> <real> <large/none>
% Unconstrained benchmark MOP

%------------------------------- Reference --------------------------------
% Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari,
% Multiobjective optimization test instances for the CEC 2009 special
% session and competition, School of CS & EE, University of Essex, Working
% Report CES-487, 2009.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            obj.M = 2;
            if isempty(obj.D); obj.D = 30; end
            obj.lower    = [0,zeros(1,obj.D-1)-1];
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            J1 = 3 : 2 : obj.D;
            J2 = 2 : 2 : obj.D;
            Y       = zeros(size(X));
            X1      = repmat(X(:,1),1,length(J1));
            Y(:,J1) = X(:,J1)-(0.3*X1.^2.*cos(24*pi*X1+4*repmat(J1,size(X,1),1)*pi/obj.D)+0.6*X1).*cos(6*pi*X1+repmat(J1,size(X,1),1)*pi/obj.D);
            X1      = repmat(X(:,1),1,length(J2));
            Y(:,J2) = X(:,J2)-(0.3*X1.^2.*cos(24*pi*X1+4*repmat(J2,size(X,1),1)*pi/obj.D)+0.6*X1).*sin(6*pi*X1+repmat(J2,size(X,1),1)*pi/obj.D); 
            PopObj(:,1) = X(:,1)         + 2*mean(Y(:,J1).^2,2);
            PopObj(:,2) = 1-sqrt(X(:,1)) + 2*mean(Y(:,J2).^2,2);
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R(:,1) = linspace(0,1,N)';
            R(:,2) = 1 - R(:,1).^0.5;
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
        	R = obj.GetOptimum(100);
        end
    end
end