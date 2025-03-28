function Fitness = FitnessWithAdaptiveAlpha2CHT(Population, alpha)
    PopCon   = sum(max(0,Population.cons),2);
    Feasible = PopCon <= 0;
    maxFeasibleValue = max(Feasible.*Population.objs);
    Fitness  = Feasible.*Population.objs + ~Feasible * maxFeasibleValue + alpha * PopCon;
end