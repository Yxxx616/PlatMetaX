function agent = DDPG_NSGAII_Alpha_Metaoptimizer(obsInfo, actInfo)
% Algorithm configuration-parameter control
% DDPG as meta-optimizer, being trained via RL.
% NSGAII as base-optimizer, with alpha setting as
% opotimization object for constrained optimization.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    agent = rlDDPGAgent(obsInfo,actInfo);
end
