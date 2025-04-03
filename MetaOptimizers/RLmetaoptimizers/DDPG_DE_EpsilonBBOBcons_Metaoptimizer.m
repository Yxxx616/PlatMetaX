function agent = DDPG_DE_EpsilonBBOBcons_Metaoptimizer(obsInfo, actInfo)
% Algorithm configuration-parameter control
% DDPG as meta-optimizer, being trained via RL.
% DE as base-optimizer, with epsilon setting as opotimization object.
% BBOB-constrained is training set.

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
