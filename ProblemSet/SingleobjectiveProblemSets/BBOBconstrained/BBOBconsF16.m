classdef BBOBconsF16 < PROBLEM
% <2022> <single> <real> <constrained/none> <learned/none>
% Linear Slope function

%------------------------------- Reference --------------------------------
% Paul Dufossé, Nikolaus Hansen, Dimo Brockhoff, Phillipe R. Sampaio, Asma 
% Atamna, and Anne Auger. Building scalable test problems for benchmarking 
% constrained optimizers. 2022. To be submitted to the SIAM Journal of 
% Optimization.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------

    properties(Access = private)
        xinit;
        xopt;   % Optimal decision variables
        fshift; % Obj shift offset 
        feasible_direction; % Feasible direction
        numberCons; 
        numberActiveCons;
        rseed; % Random seed
    end
    methods
        function Setting(obj)
            % Default settings for BBOBconsF1
            obj.M = 1; % Single objective
            if isempty(obj.D)
                obj.D = 10;
            end
            obj.xinit = zeros(1,obj.D);
            obj.numberCons = 1;
            obj.numberActiveCons = 1;
            obj.rseed = 12345; % Default random seed
            obj.lower = -5 * ones(1, obj.D);
            obj.upper = 5 * ones(1, obj.D);
            obj.xopt = generateXopt(obj.rseed, obj.D);
            obj.feasible_direction = setFeasibleDirection(obj.evaluateGradient(obj.xinit-obj.xopt), obj.xopt,obj.D,obj.rseed);
            obj.fshift = generateFopt(1,1);
            obj.encoding = ones(1, obj.D);
        end
        function PopObj = CalObj(obj, X)
            Z = X - repmat(obj.xopt, size(X, 1), 1);
            PopObj = obj.fshift + sum(Z, 2);
        end
        function PopCon = CalCon(obj, X)
            % Calculate constraint violations
            PopCon = zeros(size(X, 1), obj.numberCons);
            % Generate random factors for constraints
            rng(obj.rseed); % Set random seed for reproducibility
            global_scaling_factor = 100;
            factor1 = global_scaling_factor * 10^(rand);% First constraint using feasible_direction
            gradient_c1 = -obj.feasible_direction;
            for i = 1:obj.numberCons
                if i <= obj.numberActiveCons
                    % Active constraints
                    if i == 1
                        gradient = gradient_c1 * factor1;
                    else
                        gradient = randn(1, obj.D) * sqrt(factor1);
                    end
                    PopCon(:, i) = sum(X .* gradient, 2) - 1;
                else
                    % Inactive constraints
                    PopCon(:, i) = sum(X .* randn(1, obj.D), 2) - 1;
                end
            end
        end
        
        function grad = evaluateGradient(obj,X)
            base = 10;
            exponent = (0:obj.D-1) / (obj.D - 1);
            si = base.^exponent;
            si(obj.xopt < 0) = -si(obj.xopt < 0);
            grad = -si;
        end
    end
end
