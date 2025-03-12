classdef MLP_Alg_Rec_Baseoptimizer < BASEOPTIMIZER
% <2025> <single> <permutation> <large/none> <learned/none>
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