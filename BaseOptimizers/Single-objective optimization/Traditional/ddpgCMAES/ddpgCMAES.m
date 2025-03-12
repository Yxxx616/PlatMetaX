classdef ddpgCMAES < BASEOPTIMIZER
% <2025> <single> <real/integer> <large/none> <constrained/none>
% Covariance matrix adaptation evolution strategy

%------------------------------- Reference --------------------------------
% N. Hansen and A. Ostermeier, Completely derandomized selfadaptation in
% evolution strategies, Evolutionary Computation, 2001, 9(2): 159-195.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    properties
        curProblem
        consNum
        Population
        NP
        bestObj
        
        mu
        w
        mu_eff
        cs
        ds
        ENN
        cc
        c1
        cmu
        hth
        Mdec
        ps
        pc
        C
        sigma
    end
    methods
        function init(baseOptimizer,Problem)
            baseOptimizer.curProblem = Problem;
            baseOptimizer.NP = Problem.N;
            
            
            %% Initialization
            % Number of parents
            baseOptimizer.mu     = round(Problem.N/2);
            % Parent weights
            baseOptimizer.w      = log(baseOptimizer.mu+0.5) - log(1:baseOptimizer.mu);
            baseOptimizer.w      = baseOptimizer.w./sum(baseOptimizer.w);
            % Number of effective solutions
            baseOptimizer.mu_eff = 1/sum(baseOptimizer.w.^2);
            % Step size control parameters
            baseOptimizer.cs     = (baseOptimizer.mu_eff+2)/(Problem.D+baseOptimizer.mu_eff+5);
            baseOptimizer.ds     = 1 + baseOptimizer.cs + 2*max(sqrt((baseOptimizer.mu_eff-1)/(Problem.D+1))-1,0);
            baseOptimizer.ENN    = sqrt(Problem.D)*(1-1/(4*Problem.D)+1/(21*Problem.D^2));
            % Covariance update parameters
            baseOptimizer.cc     = (4+baseOptimizer.mu_eff/Problem.D)/(4+Problem.D+2*baseOptimizer.mu_eff/Problem.D);
            baseOptimizer.c1     = 2/((Problem.D+1.3)^2+baseOptimizer.mu_eff);
            baseOptimizer.cmu    = min(1-baseOptimizer.c1,2*(baseOptimizer.mu_eff-2+1/baseOptimizer.mu_eff)/((Problem.D+2)^2+2*baseOptimizer.mu_eff/2));
            baseOptimizer.hth    = (1.4+2/(Problem.D+1))*baseOptimizer.ENN;
            % Initialization
            baseOptimizer.Mdec  = unifrnd(Problem.lower,Problem.upper);
            baseOptimizer.ps    = zeros(1,Problem.D);
            baseOptimizer.pc    = zeros(1,Problem.D);
            baseOptimizer.C     = eye(Problem.D);
            baseOptimizer.sigma = 0.1*(Problem.upper-Problem.lower);
            baseOptimizer.Population = Problem.Initialization(1);
            baseOptimizer.bestObj = baseOptimizer.Population.obj;
        end
        
        function [reward, nextState, done, bestPop]= update(baseOptimizer,action,Problem)
            %% Optimization
            % Sample solutions
            for i = 1 : Problem.N
                Pstep(i,:) = mvnrnd(zeros(1,Problem.D),baseOptimizer.C);
            end
            Pdec       = baseOptimizer.Mdec + baseOptimizer.sigma.*Pstep;
            baseOptimizer.Population = Problem.Evaluation(Pdec);
            % Update mean
            [~,rank] = sort(FitnessEpsilonSingle(baseOptimizer.Population,action));
            Pstep    = Pstep(rank,:);
            Mstep    = baseOptimizer.w*Pstep(1:baseOptimizer.mu,:);
            baseOptimizer.Mdec     = baseOptimizer.Mdec + baseOptimizer.sigma.*Mstep;
            % Update parameters
            baseOptimizer.ps    = (1-baseOptimizer.cs)*baseOptimizer.ps + sqrt(baseOptimizer.cs*(2-baseOptimizer.cs)*baseOptimizer.mu_eff)*Mstep/chol(baseOptimizer.C)';
            baseOptimizer.sigma = baseOptimizer.sigma*exp(baseOptimizer.cs/baseOptimizer.ds*(norm(baseOptimizer.ps)/baseOptimizer.ENN-1))^0.3;
            hs    = norm(baseOptimizer.ps)/sqrt(1-(1-baseOptimizer.cs)^(2*(ceil(Problem.FE/Problem.N)+1))) < baseOptimizer.hth;
            delta = (1-hs)*baseOptimizer.cc*(2-baseOptimizer.cc);
            baseOptimizer.pc    = (1-baseOptimizer.cc)*baseOptimizer.pc + hs*sqrt(baseOptimizer.cc*(2-baseOptimizer.cc)*baseOptimizer.mu_eff)*Mstep;
            baseOptimizer.C     = (1-baseOptimizer.c1-baseOptimizer.cmu)*baseOptimizer.C + baseOptimizer.c1*(baseOptimizer.pc'*baseOptimizer.pc+delta*baseOptimizer.C);
            for i = 1 : baseOptimizer.mu
                baseOptimizer.C = baseOptimizer.C + baseOptimizer.cmu*baseOptimizer.w(i)*Pstep(i,:)'*Pstep(i,:);
            end
            [V,E] = eig(baseOptimizer.C);
            if any(diag(E)<0)
                baseOptimizer.C = V*max(E,0)/V;
            end
            
            nextState = calSOPState(baseOptimizer);
            nofinish = baseOptimizer.NotTerminated(baseOptimizer.Population);
            done = ~nofinish;
            curbest = baseOptimizer.result{end}.obj;
            if isnan(curbest)
                curbest = 0;
                reward = -1;
            elseif curbest - baseOptimizer.bestObj >=0
                reward = 0;
            else
                reward = 1;
            end
            baseOptimizer.bestObj = curbest;
            if done
                bestPop = baseOptimizer.Population;
            else
                bestPop = 0;
            end
        end
    end
end