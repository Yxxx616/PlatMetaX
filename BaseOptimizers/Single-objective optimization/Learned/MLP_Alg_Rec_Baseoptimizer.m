classdef MLP_Alg_Rec_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <permutation> <large/none> <learned/none>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    methods
        function [reward, nextState, done, bestPop] = update(this,AlgIdx,Problem)
            this.pro = Problem;
            set(0, 'DefaultFigureVisible', 'off');
            if AlgIdx == 1
                alg = ABC();
            elseif AlgIdx == 2
                alg = CSO();
            elseif AlgIdx == 3
                alg = DE();
            elseif AlgIdx == 4
                alg = PSO();
            elseif AlgIdx == 5
                alg = SA();
            end
            alg.Solve(this.pro);
            set(0, 'DefaultFigureVisible', 'on');
            this.metric = alg.metric;
            if this.pro.maxRuntime < inf
                this.pro.maxFE = this.pro.FE*this.pro.maxRuntime/this.metric.runtime;
            end
            this.result = alg.result;
            this.starttime = tic;
            
            nextState = this.pro.instanceFeatures;
            reward = -min(this.metric.Min_value); %alg.metric.runtime
            bestPop = alg.result{end};
            
            done = true;
        end
    end
end