classdef MOEADDE_F5 < PROBLEM
% <2009> <multi> <real> <large/none>
% Benchmark MOP for testing MOEA/D-DE

%------------------------------- Reference --------------------------------
% H. Li and Q. Zhang, Multiobjective optimization problems with complicated
% Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
% Computation, 2009, 13(2): 284-302.
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
            obj.lower    = [0,-ones(1,obj.D-1)];
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            J1   = 3 : 2 : obj.D;
            J2   = 2 : 2 : obj.D;
            X1J1 = repmat(X(:,1),1,length(J1));
            X1J2 = repmat(X(:,1),1,length(J2));
            PopObj(:,1) = X(:,1)         + 2*mean((X(:,J1)-(0.3*X1J1.^2.*cos(24*pi*X1J1+repmat(4*J1*pi/obj.D,size(X,1),1))+0.6*X1J1).*cos(6*pi*X1J1+repmat(J1*pi/obj.D,size(X,1),1))).^2,2);
            PopObj(:,2) = 1-sqrt(X(:,1)) + 2*mean((X(:,J2)-(0.3*X1J2.^2.*cos(24*pi*X1J2+repmat(4*J2*pi/obj.D,size(X,1),1))+0.6*X1J2).*sin(6*pi*X1J2+repmat(J2*pi/obj.D,size(X,1),1))).^2,2);
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