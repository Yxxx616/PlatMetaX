% MATLAB Code
function [offspring] = updateFunc1060(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection (top 30%)
    [~, sorted_idx] = sort(popfits, 'descend');
    elite_num = max(1, round(0.3*NP));
    elite_pop = popdecs(sorted_idx(1:elite_num),:);
    elite_weights = popfits(sorted_idx(1:elite_num)) - min(popfits) + eps;
    elite_weights = elite_weights / sum(elite_weights);
    elite_vec = elite_weights' * elite_pop;
    
    % 2. Rank-based indices with fitness weighting
    ranks = NP:-1:1;
    prob = ranks / sum(ranks);
    idx = randsample(NP, NP*4, true, prob);
    r1 = idx(1:NP);
    r2 = idx(NP+1:2*NP);
    r3 = idx(2*NP+1:3*NP);
    r4 = idx(3*NP+1:end);
    
    % 3. Constraint-aware adaptation
    max_cons = max(abs(cons)) + eps;
    alpha = 0.5 + 0.5*tanh(abs(cons)/max_cons);
    
    % 4. Adaptive scaling factors
    F1 = 0.7 + 0.1*rand(NP,1);
    F2 = 0.5 + 0.2*rand(NP,1);
    F3 = 0.3 + 0.2*rand(NP,1);
    
    % 5. Enhanced mutation
    elite_term = F1.*alpha .* (elite_vec - popdecs);
    diff_term1 = F2.*(1-alpha) .* (popdecs(r1,:) - popdecs(r2,:));
    diff_term2 = F3.*(1-alpha) .* (popdecs(r3,:) - popdecs(r4,:));
    mutants = popdecs + elite_term + diff_term1 + diff_term2;
    
    % 6. Dynamic crossover
    CR = 0.8 - 0.4*alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end