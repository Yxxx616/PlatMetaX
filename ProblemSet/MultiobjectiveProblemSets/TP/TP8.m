classdef TP8 < PROBLEM
% <2014> <multi> <real> <large/none> <robust>
% Test problem for robust multi-objective optimization
% delta --- 0.05 --- Maximum disturbance degree
% H     ---   50 --- Number of disturbances

%------------------------------- Reference --------------------------------
% A. Gaspar-Cunha, J. Ferreira, and G. Recio, Evolutionary robustness
% analysis for multi-objective optimization: benchmark problems, Structural
% and Multidisciplinary Optimization, 2014, 49: 771-793.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        delta;      % Maximum disturbance degree
        H;          % Number of disturbances
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            [obj.delta,obj.H] = obj.ParameterSet(0.05,50);
            obj.M = 2;
            if isempty(obj.D); obj.D = 5; end
            obj.lower    = [0,0,-ones(1,obj.D-2)];
            obj.upper    = [1,1, ones(1,obj.D-2)];
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(~,PopDec)
            PopObj(:,1) = PopDec(:,1);
            h = 2 - 0.8*exp(-((PopDec(:,2)-0.35)/0.25).^2) - exp(-((PopDec(:,2)-0.85)/0.03).^2);
            g = 50*sum(PopDec(:,3:end).^2,2);
            S = 1 - sqrt(PopObj(:,1));
            PopObj(:,2) = h.*(g+S);
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R(:,1) = linspace(0,1,N)';
            R(:,2) = 1 - sqrt(R(:,1));
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
        end
        %% Calculate the metric value
        function score = CalMetric(obj,metName,Population)
            switch metName
                case {'Mean_IGD','Mean_HV','Worst_IGD','Worst_HV'}
                    score = feval(metName,Population,obj);
                otherwise
                    score = feval(metName,Population,obj.optimum);
            end
        end
        %% Perturb solutions multiple times
        function PopX = Perturb(obj,PopDec,N)
            if nargin < 3; N = obj.H; end
            Delta = repmat(obj.delta.*(obj.upper-obj.lower),N*size(PopDec,1),1);
            w     = UniformPoint(N,obj.D,'Latin');
            Dec   = 2*Delta.*w(reshape(repmat(1:end,size(PopDec,1),1),1,[]),:) + repmat(PopDec,N,1) - Delta;
            Dec   = obj.CalDec(Dec);
            PopX  = SOLUTION(Dec,obj.CalObj(Dec),obj.CalCon(Dec));
        end
    end
end