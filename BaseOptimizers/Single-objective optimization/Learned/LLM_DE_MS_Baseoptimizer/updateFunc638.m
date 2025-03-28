% MATLAB Code
function [offspring] = updateFunc638(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Adaptive scaling factor based on constraints
    mean_cons = mean(cons);
    std_cons = std(cons);
    if std_cons == 0, std_cons = 1; end
    F = 0.5 * (1 + tanh((mean_cons - cons)/std_cons));
    
    % 3. Fitness-directed mutation
    mutant = zeros(NP, D);
    sigma_f = std(popfits);
    if sigma_f == 0, sigma_f = 1; end
    
    for i = 1:NP
        % Select 4 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = randsample(candidates, 4);
        
        % Calculate fitness weights
        w1 = exp(-popfits(idx(1))/sigma_f);
        w2 = exp(-popfits(idx(2))/sigma_f);
        w_sum = w1 + w2 + eps;
        
        % Weighted difference vector
        diff = (w1*(popdecs(idx(1),:) - popdecs(idx(3),:) + ...
                w2*(popdecs(idx(2),:) - popdecs(idx(4),:))/w_sum;
            
        % Combined mutation
        mutant(i,:) = popdecs(i,:) + F(i)*(elite - popdecs(i,:)) + F(i)*diff;
    end
    
    % 4. Dynamic crossover
    CR = 0.9 * (1 - tanh(abs(cons))); % Constraint-aware crossover rate
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end