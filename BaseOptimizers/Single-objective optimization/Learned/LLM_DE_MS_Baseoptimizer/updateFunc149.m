% MATLAB Code
function [offspring] = updateFunc149(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Combined score (lower is better)
    combined_score = norm_fits + norm_cons;
    
    % Weighted centroid calculation
    weights = 1./(1 + abs(norm_fits) + abs(norm_cons));
    weights = weights / sum(weights);
    centroid = weights' * popdecs;
    
    % Opposition population
    opposition = lb + ub - popdecs;
    
    % Generate random indices matrix
    rand_idx = randi(NP, NP, 5);
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3); r4 = rand_idx(:,4); r5 = rand_idx(:,5);
    
    % Adaptive parameters
    F = 0.5 * (1 + rand(NP,1)) .* (1 - norm_cons);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation strategy selection
    exploit_prob = 0.6 - 0.3 * norm_cons;
    exploit_mask = rand(NP,1) < exploit_prob;
    
    % Mutation vectors
    v = zeros(NP, D);
    
    % Exploitation component
    v(exploit_mask,:) = repmat(centroid, sum(exploit_mask), 1) + ...
                       F(exploit_mask) .* (popdecs(r1(exploit_mask),:) - popdecs(r2(exploit_mask),:)) + ...
                       F(exploit_mask) .* (opposition(r3(exploit_mask),:) - popdecs(r4(exploit_mask),:));
    
    % Exploration component
    v(~exploit_mask,:) = popdecs(r1(~exploit_mask),:) + ...
                        F(~exploit_mask) .* (popdecs(r2(~exploit_mask),:) - popdecs(r3(~exploit_mask),:)) + ...
                        F(~exploit_mask) .* (opposition(~exploit_mask,:) - popdecs(r4(~exploit_mask),:));
    
    % Crossover
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary handling with opposition-based learning
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % For feasible solutions, use opposition-based reflection
    feasible = norm_cons < 0.5;
    reflect_mask = repmat(feasible, 1, D) & ((offspring < lb_rep) | (offspring > ub_rep));
    offspring(reflect_mask) = opposition(reflect_mask);
    
    % For infeasible solutions, use random reinitialization
    infeasible_mask = ~feasible;
    reinit_mask = repmat(rand(NP,1) < norm_cons, 1, D) & ((offspring < lb_rep) | (offspring > ub_rep));
    offspring(reinit_mask) = lb_rep(reinit_mask) + rand(sum(reinit_mask(:)),1) .* (ub_rep(reinit_mask) - lb_rep(reinit_mask));
    
    % For remaining violations, use midpoint between current and centroid
    remaining_violations = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(remaining_violations) = (offspring(remaining_violations) + ...
                                     repmat(centroid, NP, 1)(remaining_violations)) / 2;
end