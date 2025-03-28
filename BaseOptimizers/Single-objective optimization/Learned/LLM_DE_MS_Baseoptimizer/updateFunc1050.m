% MATLAB Code
function [offspring] = updateFunc1050(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with exponential weighting
    [sorted_f, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.3*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Exponential weights based on rank
    ranks = (1:elite_num)';
    alpha = 0.5;
    weights = exp(-alpha * ranks);
    weights = weights / sum(weights);
    
    % Weighted elite vector (vectorized)
    elite_vec = sum(weights .* elite_pool, 1);
    
    % 2. Adaptive scaling factors
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    norm_f = (popfits - f_mean) / f_std;
    F1 = 0.5 * (1 + tanh(norm_f));
    
    norm_c = abs(cons) / (max(abs(cons)) + eps);
    F2 = 0.3 * (1 - exp(-norm_c));
    F3 = 0.2 * norm_c;
    
    % 3. Constraint-aware perturbation
    xi = sign(cons) .* randn(NP, 1);
    xi = xi * ones(1, D) .* randn(NP, D);
    
    % 4. Differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Combined mutation (vectorized)
    elite_diff = elite_vec - popdecs;
    mutants = popdecs + F1.*elite_diff + F2.*diff_vec + F3.*xi;
    
    % 6. Dynamic crossover
    CR = 0.9 * (1 - norm_c); % More exploration for constrained solutions
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Smart boundary handling
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    
    % Reflection with adaptive scaling
    reflect_lb = 2*lb - offspring;
    reflect_ub = 2*ub - offspring;
    
    % Blend with random component based on fitness
    rnd_scale = 0.1 + 0.4*(1 - norm_f);
    rand_comp = lb + (ub-lb).*rand(NP,D);
    
    offspring(lb_viol) = (1-rnd_scale(lb_viol(:,1))).*reflect_lb(lb_viol) + ...
                        rnd_scale(lb_viol(:,1)).*rand_comp(lb_viol);
    offspring(ub_viol) = (1-rnd_scale(ub_viol(:,1))).*reflect_ub(ub_viol) + ...
                        rnd_scale(ub_viol(:,1)).*rand_comp(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end