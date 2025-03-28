% MATLAB Code
function [offspring] = updateFunc228(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint processing with sigmoid normalization
    abs_cons = abs(cons);
    max_con = max(abs_cons) + eps;
    min_con = min(abs_cons);
    weights = 1 ./ (1 + exp(5 * (abs_cons - min_con) ./ (max_con - min_con + eps)));
    
    % Fitness normalization
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - avg_fit) ./ std_fit;
    
    % Population diversity measure
    diversity = mean(std(popdecs)) + eps;
    
    % Elite selection considering both fitness and constraints
    combined_scores = popfits + 1e6*abs_cons;
    [~, elite_idx] = min(combined_scores);
    elite = popdecs(elite_idx, :);
    
    % Best feasible solution selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible,:);
        best_feas = best_feas(best_feas_idx,:);
    else
        [~, min_con_idx] = min(abs_cons);
        best_feas = popdecs(min_con_idx,:);
    end
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 - tanh(norm_fits));  % Fitness-based
    F2 = 0.3 * (1 - weights);          % Constraint-based
    F3 = 0.2 * diversity/D;            % Diversity-based
    
    % Vectorized mutation
    for i = 1:NP
        % Select two distinct random individuals
        idxs = randperm(NP, 2);
        while any(idxs == i)
            idxs = randperm(NP, 2);
        end
        
        % Elite-directed differential component
        diff1 = popdecs(idxs(1),:) - popdecs(idxs(2),:);
        
        % Constraint-aware perturbation
        diff2 = best_feas - popdecs(i,:);
        
        % Combine components with adaptive weights
        offspring(i,:) = elite + F1(i)*diff1 + F2(i)*diff2 + F3*randn(1,D);
    end
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + rand(NP,D).*(lb - offspring)).*out_low + ...
               (ub - rand(NP,D).*(offspring - ub)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end