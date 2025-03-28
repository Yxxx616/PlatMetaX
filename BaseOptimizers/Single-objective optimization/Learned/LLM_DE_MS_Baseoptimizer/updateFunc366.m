% MATLAB Code
function [offspring] = updateFunc366(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility ratio
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraint violations
    min_cons = min(cons);
    max_cons = max(cons);
    phi = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Normalize fitness values
    min_fit = min(popfits);
    max_fit = max(popfits);
    psi = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Combined weights
    w = rho * psi + (1 - rho) * (1 - phi);
    
    % Select elite and leader
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx, :);
    
    if any(feasible_mask)
        [~, leader_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        leader = popdecs(temp(leader_idx), :);
    else
        [~, leader_idx] = min(cons);
        leader = popdecs(leader_idx, :);
    end
    
    % Generate mutation vectors
    F1 = 0.5 * w;
    F2 = 0.5 * (1 - w);
    diff_leader = bsxfun(@minus, leader, popdecs);
    diff_elite = bsxfun(@minus, elite, popdecs);
    
    offspring = popdecs + bsxfun(@times, F1, diff_leader) + ...
                         bsxfun(@times, F2, diff_elite);
    
    % Add adaptive perturbation
    sigma = 0.2 * (1 - w);
    offspring = offspring + bsxfun(@times, sigma, randn(NP, D));
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring .* (~out_low & ~out_high) + ...
        (2 * lb - offspring) .* out_low + ...
        (2 * ub - offspring) .* out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end