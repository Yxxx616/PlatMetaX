% MATLAB Code
function [offspring] = updateFunc1014(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize objectives and constraints
    f_norm = (popfits - mean(popfits)) / (std(popfits) + 1e-12);
    c_norm = (cons - mean(cons)) / (std(cons) + 1e-12);
    
    % Combined adaptive weights
    w = 1./(1 + exp(-5*(f_norm + c_norm)));
    w = w / max(w);
    
    % Elite vector (weighted centroid)
    x_elite = sum(popdecs .* w, 1) / sum(w);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Constraint-adaptive scaling
    F = 0.5 * (1 + tanh(5*c_norm));
    
    % Mutation with elite guidance
    elite_term = x_elite - popdecs;
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    d = w.*elite_term + (1-w).*diff_term;
    noise = 0.1 * (1-w) .* randn(NP, D);
    mutants = popdecs + F.*d + noise;
    
    % Opposition-based refinement
    opp_mask = rand(NP,1) < 0.1*(1-w);
    mutants(opp_mask,:) = lb + ub - mutants(opp_mask,:);
    
    % Dynamic crossover
    CR = 0.9*w + 0.1;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end