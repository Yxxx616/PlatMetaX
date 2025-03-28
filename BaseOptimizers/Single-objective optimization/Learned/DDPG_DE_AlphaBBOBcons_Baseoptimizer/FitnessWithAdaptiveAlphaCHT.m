function Fitness = FitnessWithAdaptiveAlphaCHT(Population, alpha)
    PopCon   = sum(max(0,Population.cons),2);
    Fitness  = Population.objs + alpha*PopCon;
end