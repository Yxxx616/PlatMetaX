% MATLAB Code
function [offspring] = updateFunc148(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    min_con = min(cons);
    max_con = max(cons);
    norm_cons = (cons - min_con) / (max_con - min_con + eps);
    
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Combined score (lower is better)
    combined_score = norm_fits + norm_cons;
    
    % Identify best, elite and poor individuals
    [~, sorted_idx] = sort(combined_score);
    elite_size = max(2, floor(0.2 * NP));
    elite_idx = sorted_idx(1:elite_size);
    poor_idx = sorted_idx(end-elite_size+1:end);
    
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_poor = mean(popdecs(poor_idx,:), 1);
    [~, best_idx] = min(combined_score);
    x_best = popdecs(best_idx,:);
    
    % Generate random indices matrix (vectorized)
    rand_idx = randi(NP, NP, 4);
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3);
    r4 = rand_idx(:,4);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - norm_cons);
    CR = 0.9 - 0.4 * norm_cons;
    
    % Mutation strategies based on population quality
    v = zeros(NP, D);
    is_elite = ismember(1:NP, elite_idx)';
    is_poor = ismember(1:NP, poor_idx)';
    mid_mask = ~is_elite & ~is_poor;
    
    % Elite individuals - exploitation
    v(is_elite,:) = repmat(x_elite, sum(is_elite), 1) + ...
                   F(is_elite) .* (repmat(x_elite - x_poor, sum(is_elite), 1) + ...
                   popdecs(r1(is_elite),:) - popdecs(r2(is_elite),:));
    
    % Middle population - balanced exploration/exploitation
    v(mid_mask,:) = popdecs(mid_mask,:) + ...
                   (1 + 0.5 * norm_cons(mid_mask)) .* F(mid_mask) .* ...
                   (popdecs(r1(mid_mask),:) - popdecs(r2(mid_mask),:)) + ...
                   F(mid_mask) .* (popdecs(r3(mid_mask),:) - popdecs(r4(mid_mask),:));
    
    % Poor individuals - exploration
    v(is_poor,:) = popdecs(r1(is_poor),:) + ...
                  F(is_poor) .* (popdecs(r2(is_poor),:) - popdecs(r3(is_poor),:)) + ...
                  F(is_poor) .* (popdecs(r4(is_poor),:) - popdecs(is_poor,:));
    
    % Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % For feasible solutions, use reflection
    feasible = norm_cons < 0.1;
    reflect_mask = repmat(feasible, 1, D) & ((offspring < lb_rep) | (offspring > ub_rep));
    offspring(reflect_mask & (offspring < lb_rep)) = 2*lb_rep(reflect_mask & (offspring < lb_rep)) - offspring(reflect_mask & (offspring < lb_rep));
    offspring(reflect_mask & (offspring > ub_rep)) = 2*ub_rep(reflect_mask & (offspring > ub_rep)) - offspring(reflect_mask & (offspring > ub_rep));
    
    % For infeasible solutions, use random reinitialization with probability
    infeasible_mask = ~feasible;
    reinit_prob = norm_cons ./ max(norm_cons);
    reinit_mask = repmat(rand(NP,1) < reinit_prob, 1, D) & ((offspring < lb_rep) | (offspring > ub_rep));
    rand_vals = lb_rep(reinit_mask) + rand(sum(reinit_mask(:)),1) .* (ub_rep(reinit_mask) - lb_rep(reinit_mask));
    offspring(reinit_mask) = rand_vals;
    
    % For remaining violations, use midpoint reflection
    remaining_violations = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(remaining_violations & (offspring < lb_rep)) = (lb_rep(remaining_violations & (offspring < lb_rep)) + x_elite(remaining_violations & (offspring < lb_rep))) / 2;
    offspring(remaining_violations & (offspring > ub_rep)) = (ub_rep(remaining_violations & (offspring > ub_rep)) + x_elite(remaining_violations & (offspring > ub_rep))) / 2;
end