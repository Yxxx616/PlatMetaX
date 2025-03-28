% MATLAB Code
function [offspring] = updateFunc637(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        elite = popdecs(min_cons_idx, :);
    end
    
    % 2. Compute adaptive parameters
    mean_fit = mean(popfits);
    std_fit = std(popfits);
    if std_fit == 0, std_fit = 1; end
    T = std_fit; % Temperature for weights
    
    % 3. Generate mutation vectors
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select distinct indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        
        % Compute fitness weights
        f_weights = exp(-popfits(idx(1:2))/T);
        f_weights = f_weights / sum(f_weights);
        
        % Create weighted difference
        diff_vec = f_weights(1)*(popdecs(idx(1),:) - popdecs(idx(3),:)) + ...
                   f_weights(2)*(popdecs(idx(2),:) - popdecs(idx(4),:));
        
        % Constraint-aware perturbation
        pert_mag = tanh(abs(cons(i))) * randn(1,D) .* mean(abs(elite - popdecs(i,:)));
        
        % Combine components
        F = 0.5 + 0.3*randn();
        mutant(i,:) = popdecs(i,:) + F*(elite - popdecs(i,:)) + ...
                      F*diff_vec + pert_mag;
    end
    
    % 4. Adaptive crossover
    norm_fits = (popfits - mean_fit)/std_fit;
    CR = 0.5 * (1 + erf(norm_fits/sqrt(2))) .* (1 - tanh(abs(cons)));
    CR = max(min(CR, 0.95), 0.05);
    
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    % 5. Create offspring
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for out-of-bounds
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clipping
    offspring = max(min(offspring, ub_rep), lb_rep);
end