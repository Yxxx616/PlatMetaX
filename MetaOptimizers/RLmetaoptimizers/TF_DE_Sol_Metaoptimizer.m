function agent =  TF_DE_Sol_Metaoptimizer(obsInfo, actInfo)
% to do %
% Algorithm generation-solution manipulation
% required MATLAB >= 2024a
% Transformer neural network as meta-policy, being trained via RL.
% DE as base-optimizer, with solution manipulation as opotimization object.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    initoption = rlAgentInitializationOptions(UseRNN=true);
    agentoption = rlDDPGAgentOptions(SequenceLength=48);
    actorNetwork = createActorTransformer(obsInfo, actInfo);
    actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo,'ObservationInputNames','input');

    criticNetwork = createCriticNetwork(obsInfo, actInfo);
    critic = rlQValueFunction(criticNetwork,obsInfo,actInfo,...
        ObservationInputNames="obsInput", ...
        ActionInputNames="actionInput");

    agent = rlDDPGAgent(actor, critic);
end
    
function net = createActorTransformer(obsInfo,actInfo)
    % 构建Transformer网络架构
    inputDim = obsInfo.Dimension(1);  
    outputDim = actInfo.Dimension(1);
                
    layers = [
        sequenceInputLayer(inputDim, 'Name', 'input')  % 输入层
        positionEmbeddingLayer(inputDim, obsInfo.Dimension(2), 'Name', 'pos_emb')
        additionLayer(2,Name="add")
        % 编码器部分 (重复两次)
        % 编码器块1
        selfAttentionLayer(8, 256, 'Name', 'enc_self_attn1')  % 维度保持256
        additionLayer(2, 'Name', 'enc_add1')                  
        layerNormalizationLayer('Name', 'enc_ln1')            
        fullyConnectedLayer(256*2, 'Name', 'enc_fc1')         % 扩展维度到512
        reluLayer('Name', 'enc_relu1')                        
        fullyConnectedLayer(inputDim, 'Name', 'enc_fc2')           % 恢复维度到inputDim
        additionLayer(2, 'Name', 'enc_add2')                  
        layerNormalizationLayer('Name', 'enc_ln2')            
        
        % 编码器块2 (重复结构)
        selfAttentionLayer(8, 256, 'Name', 'enc_self_attn2')  % 维度保持256
        additionLayer(2, 'Name', 'enc_add3')
        layerNormalizationLayer('Name', 'enc_ln3')
        fullyConnectedLayer(256*2, 'Name', 'enc_fc3')
        reluLayer('Name', 'enc_relu2')
        fullyConnectedLayer(inputDim, 'Name', 'enc_fc4')          % 恢复维度到256
        additionLayer(2, 'Name', 'enc_add4')
        layerNormalizationLayer('Name', 'enc_ln4')
        
        % 解码器部分
        % 解码器自注意力
        selfAttentionLayer(8, 256, 'AttentionMask', 'causal', 'Name', 'dec_self_attn')
        additionLayer(2, 'Name', 'dec_add1')
        layerNormalizationLayer('Name', 'dec_ln1')
        
        % 交叉注意力（编码器->解码器）
        attentionLayer(4, 'Name', 'dec_cross_attn')  
        layerNormalizationLayer('Name', 'dec_ln2')
        
        % 前馈网络（带维度匹配）
        fullyConnectedLayer(256*2, 'Name', 'dec_fc1')
        reluLayer('Name', 'dec_relu1')
        fullyConnectedLayer(inputDim, 'Name', 'dec_fc2')          % 恢复维度到256
        additionLayer(2, 'Name', 'dec_add3')
        layerNormalizationLayer('Name', 'dec_ln3')
        
        % 输出部分（新增完整输出结构）
        fullyConnectedLayer(256, 'Name', 'output_fc1')        % 过渡层
        reluLayer('Name', 'output_relu')
        fullyConnectedLayer(outputDim, 'Name', 'final_fc')
        sigmoidLayer('Name', 'prob_output')                   % 输出层
    ];
    
    % 创建 layerGraph 并连接
    lgraph = layerGraph(layers);
    
    %% 编码器
    lgraph = connectLayers(lgraph, 'input', 'add/in2');
    lgraph = connectLayers(lgraph, 'input', 'enc_add1/in2');   % 跳跃连接输入
    
    % 第一层前馈残差（确保维度匹配）
    lgraph = connectLayers(lgraph, 'enc_ln1', 'enc_add2/in2');
    
    % 第二层自注意力残差
    lgraph = connectLayers(lgraph, 'enc_ln2', 'enc_add3/in2');
    
    % 第二层前馈残差
    lgraph = connectLayers(lgraph, 'enc_ln3', 'enc_add4/in2');
    
    %% 解码器连接修正
    % 自注意力残差
    lgraph = connectLayers(lgraph, 'input', 'dec_add1/in2');  % 假设共享输入维度
    
    % 交叉注意力连接
    lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/key');
    lgraph = connectLayers(lgraph, 'enc_ln4', 'dec_cross_attn/value');
    
    % 前馈残差连接
    lgraph = connectLayers(lgraph, 'dec_ln2', 'dec_add3/in2');
    
    % analyzeNetwork(lgraph)
    % 创建可训练网络
    net = dlnetwork(lgraph);
end

function criticNet = createCriticNetwork(obsInfo, actInfo)
% Define observation and action paths
obsPath = sequenceInputLayer(obsInfo.Dimension, 'Name', 'obsInput');
actPath = sequenceInputLayer(actInfo.Dimension, 'Name', 'actionInput');

% Define common path: concatenate along first dimension.
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(1)
    ];

criticNet = dlnetwork;
criticNet = addLayers(criticNet, obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);

criticNet = connectLayers(criticNet,"obsInput","concat/in1");
criticNet = connectLayers(criticNet,"actionInput","concat/in2");
criticNet = initialize(criticNet);
end