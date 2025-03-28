function Fitness = FitnessWithAdaptiveEpsilonCHT(Population, epsilon)
    PopCon   = sum(max(0,Population.cons),2);
    Feasible = PopCon <= epsilon;
    Fitness  = Feasible.*Population.objs + ~Feasible.*(PopCon+1e10);
end