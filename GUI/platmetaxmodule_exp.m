classdef platmetaxmodule_exp < handle
%module_exp - Experimental module.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        platmetaxGUI;                % The platmetaxGUI object
        app  = struct();	% All the components
        data = [];          % All the results
    end
    methods(Access = ?platmetaxGUI)
        %% Constructor
        function obj = platmetaxmodule_exp(platmetaxGUI)
            % The main grid
            obj.platmetaxGUI = platmetaxGUI;
            obj.app.maingrid = platmetaxGUI.APP(3,1,uigridlayout(obj.platmetaxGUI.app.maingrid,'RowHeight',{'1x'},'ColumnWidth',{'4x','1.2x'},'Padding',[0 5 0 5],'RowSpacing',5,'ColumnSpacing',0,'BackgroundColor','w'));
            

            % The first panel
            obj.app.grid(3)    = platmetaxGUI.APP(1,[1 4],uigridlayout(obj.app.maingrid,'RowHeight',{20,30,'1x',30},'ColumnWidth',{230,'1x','1x',230},'Padding',[20 10 20 0],'ColumnSpacing',20,'BackgroundColor','w'));
            obj.app.label(3) = platmetaxGUI.APP(1,[2 3],uilabel(obj.app.grid(3),'Text','Result display','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            tempGrid           = platmetaxGUI.APP(2,[1 4],uigridlayout(obj.app.grid(3),'RowHeight',{1,'1x',1},'ColumnWidth',{18,24,24,24,24,24,'1x','1x','1x','1.2x'},'Padding',[5 5 5 5],'RowSpacing',0,'ColumnSpacing',8,'BackgroundColor',[.95 .95 1]));
            tempPanel          = platmetaxGUI.APP(2,1,uipanel(tempGrid,'BorderType','none','BackgroundColor',[.95 .95 1]));
            obj.app.toolC(1)   = uibutton(tempPanel,'Position',[-2.5 -2.5 24 24],'Text','','Icon',obj.platmetaxGUI.icon.savetable,'BackgroundColor',[.95 .95 1],'Tooltip','Save the table','ButtonpushedFcn',@obj.cb_save);
            tempPanel          = platmetaxGUI.APP(2,2,uipanel(tempGrid,'BorderType','none','BackgroundColor',[.95 .95 1]));
            obj.app.toolC(2)   = uibutton(tempPanel,'Position',[-2.5 -2.5 31 24],'Text','','Icon',obj.platmetaxGUI.icon.figure,'BackgroundColor',[.95 .95 1],'Tooltip','Display the results','ButtonpushedFcn',@obj.cb_tableDisplay);
            obj.app.toolC(3)   = platmetaxGUI.APP([1 3],3,uibutton(tempGrid,'state','Text','N','BackgroundColor',[.95 .95 1],'Tooltip','Show the population size','ValueChangedFcn',@obj.TableUpdateColumn));
            obj.app.toolC(4)   = platmetaxGUI.APP([1 3],4,uibutton(tempGrid,'state','Text','M','BackgroundColor',[.95 .95 1],'Value',1,'Tooltip','Show the number of objectives','ValueChangedFcn',@obj.TableUpdateColumn));
            obj.app.toolC(5)   = platmetaxGUI.APP([1 3],5,uibutton(tempGrid,'state','Text','D','BackgroundColor',[.95 .95 1],'Value',1,'Tooltip','Show the number of decision variables','ValueChangedFcn',@obj.TableUpdateColumn));
            obj.app.toolC(6)   = platmetaxGUI.APP([1 3],6,uibutton(tempGrid,'state','Text','FE','BackgroundColor',[.95 .95 1],'Tooltip','Show the maximum number of function evaluations','ValueChangedFcn',@obj.TableUpdateColumn));
            obj.app.dropC(1)   = platmetaxGUI.APP([1 3],7,uidropdown(tempGrid,'BackgroundColor',[.95 .95 1],'Tooltip','Show the specific metric values','Items',{},'Interruptible','off','BusyAction','cancel','ValueChangedFcn',@obj.TableUpdate));
            obj.app.dropC(2)   = platmetaxGUI.APP([1 3],8,uidropdown(tempGrid,'BackgroundColor',[.95 .95 1],'Tooltip','Show the mean value and standard deviation','Items',{'Mean','Mean (STD)','Median','Median (IQR)'},'ItemsData',1:4,'Value',2,'Interruptible','off','BusyAction','cancel','ValueChangedFcn',@obj.TableUpdate));
            obj.app.dropC(3)   = platmetaxGUI.APP([1 3],9,uidropdown(tempGrid,'BackgroundColor',[.95 .95 1],'Tooltip','Perform the Wilcoxon rank sum test','Items',{'none','Signed rank test','Rank sum test','Friedman test'},'ItemsData',1:4,'Value',3,'Interruptible','off','BusyAction','cancel','ValueChangedFcn',@obj.TableUpdate));
            obj.app.dropC(4)   = platmetaxGUI.APP([1 3],10,uidropdown(tempGrid,'BackgroundColor',[.95 .95 1],'Tooltip','Highlight the best result','Items',{'none','Highlight the best','Highlight all the bests'},'ItemsData',1:3,'Value',2,'Interruptible','off','BusyAction','cancel','ValueChangedFcn',@obj.TableUpdate));
            obj.app.table      = platmetaxGUI.APP(3,[1 4],uitable(obj.app.grid(3),'CellSelectionCallback',@obj.cb_tableSelect));
            
            obj.app.tablemenu  = uicontext(obj.platmetaxGUI.app.figure,110);
            obj.app.tablemenu.add('  Populations (obj.)','',{@obj.cb_tableShow,1});
            obj.app.tablemenu.add('  Populations (dec.)','',{@obj.cb_tableShow,2});
            obj.app.tablemenu.add('  Metric values','',{@obj.cb_tableShow,3});
            obj.app.tablemenu.flush();

            % The second panel
            obj.app.grid(1)    = platmetaxGUI.APP(1,[5 10],uigridlayout(obj.app.maingrid,'RowHeight',{20,16,21,16,21,16,21,20,'1x',20,'1x',21,30},'ColumnWidth',{'1x','1x'},'Padding',[8 5 8 0],'RowSpacing',3,'ColumnSpacing',5,'BackgroundColor','w'));
            obj.app.label(1) = platmetaxGUI.APP(1,[1 9],uilabel(obj.app.grid(1),'Text','Algorithm selection','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            [obj.app.stateA,obj.app.labelA] = platmetaxGUI.GenerateLabelButtonTest(obj.app.grid(1),[0 1 0 1,zeros(1,13)],@obj.cb_filter);
            obj.app.labelA(4)  = platmetaxGUI.APP(8,[1 2],uilabel(obj.app.grid(1),'Text','BaseOptimizers','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.labelA(5)  = platmetaxGUI.APP(8,5,uilabel(obj.app.grid(1),'HorizontalAlignment','right','FontSize',10,'FontColor',[0.93,0.69,0.13]));
            obj.app.listA(1)   = platmetaxGUI.APP(9,[1 5],uilistbox(obj.app.grid(1),'FontColor',[0.93,0.69,0.13]));
            obj.app.labelA(6)  = platmetaxGUI.APP(8,[6 7],uilabel(obj.app.grid(1),'Text','Problems','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.labelA(7)  = platmetaxGUI.APP(8,10,uilabel(obj.app.grid(1),'HorizontalAlignment','right','FontSize',10,'FontColor',[0.30,0.75,0.93]));
            obj.app.listA(2)   = platmetaxGUI.APP(9,[6 10],uilistbox(obj.app.grid(1),'FontColor',[0.30,0.75,0.93]));
            obj.app.dropA(1)   = platmetaxGUI.APP(8,[3 4],uidropdown(obj.app.grid(1),'BackgroundColor','w','FontColor',[0.93,0.69,0.13],'Items',{'All year'},'ValueChangedFcn',@(h,~)platmetaxGUI.UpdateAlgProListYear(obj.app.listA(1),h,obj.app.labelA(5),obj.platmetaxGUI.algList)));
            obj.app.dropA(2)   = platmetaxGUI.APP(8,[7 8],uidropdown(obj.app.grid(1),'BackgroundColor','w','FontColor',[0.30,0.75,0.93],'Items',{'All year'},'ValueChangedFcn',@(h,~)platmetaxGUI.UpdateAlgProListYear(obj.app.listA(2),h,obj.app.labelA(7),obj.platmetaxGUI.proList)));
            
            
            % The third panel
            obj.app.label(2) = platmetaxGUI.APP(10,[1 9],uilabel(obj.app.grid(1),'Text','Parameter setting','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.listB   = uilist(obj.app.grid(1),obj.platmetaxGUI.app.figure,obj.platmetaxGUI.icon);
            obj.app.grid(2) = platmetaxGUI.APP(11,[1 10],obj.app.listB.grid);
            obj.app.listA(1).ValueChangedFcn = @(~,~)platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(1),obj.app.listB,'BASEOPTIMIZER',2);
            obj.app.listA(2).ValueChangedFcn = @(~,~)platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(2),obj.app.listB,'PROBLEM',-2);

            tempGridNew1       = platmetaxGUI.APP(12,[1 10],uigridlayout(obj.app.grid(1),'RowHeight',{'1x'},'ColumnWidth',{'0.9x','0.4x','1x','0.4x','0.4x','0.2x','0.8x'},'Padding',[0 0 0 0],'RowSpacing',0,'ColumnSpacing',5,'BackgroundColor','w'));
            obj.app.labelA(8)  = platmetaxGUI.APP(1,1,uilabel(tempGridNew1,'Text','Number of runs','FontColor',[0.00,0.00,0.00],'FontWeight','bold','Tooltip','Number of runs for each algorithm on each problem'));
            obj.app.editA(1)   = platmetaxGUI.APP(1,2,uieditfield(tempGridNew1,'numeric','Value',31,'limits',[1 inf],'RoundFractionalValues','on','Tooltip','Number of runs for each algorithm on each problem'));
            obj.app.labelA(9)  = platmetaxGUI.APP(1,3,uilabel(tempGridNew1,'Text','Number of results','FontColor',[0.00,0.00,0.00],'FontWeight','bold','Tooltip','Number of populations saved in each run'));
            obj.app.editA(2)   = platmetaxGUI.APP(1,4,uieditfield(tempGridNew1,'numeric','Value',10,'limits',[1 inf],'RoundFractionalValues','on','Tooltip','Number of populations saved in each run'));
            tempGrid           = platmetaxGUI.APP(1,[5 8],uigridlayout(tempGridNew1,'RowHeight',{'1x'},'ColumnWidth',{'0.5x',20,'1x'},'Padding',[0 0 0 0],'RowSpacing',0,'ColumnSpacing',5,'BackgroundColor','w'));
            obj.app.labelA(10) = platmetaxGUI.APP(1,1,uilabel(tempGrid,'Text','File path','FontColor',[0.00,0.00,0.00],'FontWeight','bold','Tooltip','File path for saving experimental settings'));
            obj.app.buttonA    = platmetaxGUI.APP(1,2,uibutton(tempGrid,'Text','...','BackgroundColor','w','ButtonpushedFcn',@obj.cb_filepath,'Tooltip','File path for saving experimental settings'));
            obj.app.editA(3)   = platmetaxGUI.APP(1,[3 4],uieditfield(tempGrid,'Value',fullfile(cd,'Data','Setting.mat'),'HorizontalAlignment','right','Tooltip','File path for saving experimental settings'));
            
%             obj.app.checkC     = platmetaxGUI.APP(13,[1 2],uicheckbox(obj.app.grid(1),'Text','Parallel execution','Tooltip','Perform the experiment with multiple CPUs','Enable',~isempty(ver('parallel'))));
            obj.app.buttonC(1) = platmetaxGUI.APP(13,[4 5],uibutton(obj.app.grid(1),'push','Text','Start','FontSize',16,'FontColor',[1 1 1],"BackgroundColor",[0.07,0.62,1.00],'ButtonpushedFcn',@obj.cb_start));
            obj.app.buttonC(2) = platmetaxGUI.APP(13,[6 7],uibutton(obj.app.grid(1),'push','Text','Stop','FontSize',16,'FontColor',[1 1 1],"BackgroundColor",[0.07,0.62,1.00],'Enable','off','ButtonpushedFcn',@obj.cb_stop));
            obj.app.labelC     = platmetaxGUI.APP(13,8,uilabel(obj.app.grid(1),'Text','','HorizontalAlignment','right','VerticalAlignment','center'));
            % Initialization
            obj.cb_filter([],[],2);
        end
    end
    methods(Access = private)
        %% Update the algorithms and problems in the lists
        function cb_filter(obj,~,~,index)
            % Update the lists of algorithms and problems
            func = platmetaxGUI.UpdateAlgProList(index,obj.app.stateA,obj.app.listA(1),obj.app.dropA(1),obj.app.labelA(5),obj.platmetaxGUI.algList,obj.app.listA(2),obj.app.dropA(2),obj.app.labelA(7),obj.platmetaxGUI.proList);
            % Update the list of metrics
            show = cellfun(@(s)func(s(2:end,1:end-2)),obj.platmetaxGUI.metList(:,1));
            obj.app.dropC(1).Items = ['Number of runs';'runtime';obj.platmetaxGUI.metList(show,2)];
            obj.TableUpdate();
        end
        %% Load or save experimental settings
        function success = cb_filepath(obj,~,~,filename)
            success = false;
            if nargin < 4   % Load experimental settings
                [file,folder] = uigetfile({'*.mat','MAT file'},'',fileparts(obj.app.editA(3).Value));
                figure(obj.platmetaxGUI.app.figure);
                if ischar(file)
                    try
                        filename = fullfile(folder,file);
                        load(filename,'Setting','Environment','-mat');
                        obj.app.listB.del(1:length(obj.app.listB.items));
                        cellfun(@(s)obj.app.listB.add([s,'.m'],2),Setting{1});
                        cellfun(@(s)obj.app.listB.add([s,'.m'],-2),Setting{2});
                        set([obj.app.listB.items.edit],{'Value'},Setting{3});
                        obj.app.listB.flush();
                        set(obj.app.editA(1:2),{'Value'},Environment);
                        obj.app.editA(3).Value = filename;
                    catch
                        uialert(obj.platmetaxGUI.app.figure,sprintf('Fail to load the experimental settings from %s.',filename),'Error');
                        return;
                    end
                end
            else            % Save experimental settings
                index       = find([obj.app.listB.items.type]<0,1);
                Setting{1}  = get([obj.app.listB.items(1:index-1).title],{'Text'});
                Setting{2}  = get([obj.app.listB.items(index:end).title],{'Text'});
                Setting{3}  = get([obj.app.listB.items.edit],{'Value'});
                Environment = get([obj.app.editA(1:2)],{'Value'});
                try
                    [folder,file] = fileparts(filename);
                    if isempty(file)
                        file = 'Setting';
                    end
                    filename = fullfile(folder,[file,'.mat']);
                    [~,~]    = mkdir(folder);
                    save(filename,'Setting','Environment','-mat');
                    obj.app.editA(3).Value = filename;
                catch
                    uialert(obj.platmetaxGUI.app.figure,sprintf('Fail to save the experimental settings to %s.',filename),'Error');
                    return;
                end
            end
            success = true;
        end
        %% Start the execution
        function cb_start(obj,~,~)
            if strcmp(obj.app.buttonC(1).Text,'Pause')
                obj.app.buttonC(1).Text = 'Continue';
            elseif strcmp(obj.app.buttonC(1).Text,'Continue')
                obj.app.buttonC(1).Text = 'Pause';
            else
                try
                    % Check the validity of settings
%                     20250331修改，去掉勾选Parallel execution的逻辑，防止用户勾选Parallel execution运行报错
%                     isParallel = obj.app.checkC.Value;
                    PRO = [];
                    assert(~isempty(obj.app.listB.items),'No algorithm is selected.');
                    type = [obj.app.listB.items.type];
                    assert(any(type>0),'No algorithm is selected.');
                    assert(any(type<0),'No problem is selected.');
                    allList  = [obj.platmetaxGUI.algList;obj.platmetaxGUI.proList];
                    allLabel = any(cell2mat(allList(ismember(allList(:,2),get([obj.app.listB.items.title],'Text')),1)),1);
                    SorM     = [allLabel(2),allLabel(3)||allLabel(4)];
                    assert(SorM(1)~=SorM(2),'Cannot perform single- and multi-objective optimization simultaneously.');
                    % Generate the BASEOPTIMIZER and PROBLEM objects
                    for i = 1 : length(type)
                        [name,para] = platmetaxGUI.GetParameterSetting(obj.app.listB.items(i));
                        if type(i) > 0
                            %                     20250331修改，去掉勾选Parallel execution的逻辑，防止用户勾选Parallel execution运行报错
%                             if ~isParallel
                                ALG(i).alg = feval(name,'parameter',para,'save',obj.app.editA(2).Value,'outputFcn',@obj.outputFcn);
%                             else
%                                 ALG(i).alg = feval(name,'parameter',para,'save',obj.app.editA(2).Value,'outputFcn',@(~,~)[]);
%                             end
                        else
                            len = cellfun(@length,para(1:4));
                            for j = 1 : max(1,max(len))
                                paraSub        = para;
                                paraSub(len>1) = cellfun(@(s)s(min(end,j)),para(len>1),'UniformOutput',false);
                                PRO            = [PRO,feval(name,'N',paraSub{1},'M',paraSub{2},'D',paraSub{3},obj.app.listB.items(i).label(4).Text,paraSub{4},'parameter',paraSub(5:end))];
                            end
                        end
                    end
                catch err
                    uialert(obj.platmetaxGUI.app.figure,err.message,'Invalid parameter settings');
                    return;
                end
                % Save the experimental settings
                if ~obj.cb_filepath([],[],obj.app.editA(3).Value)
                    return;
                end
                obj.data = struct('ALG',ALG,'PRO',PRO,'folder',fileparts(obj.app.editA(3).Value),'result',{cell(length(PRO),length(ALG),obj.app.editA(1).Value)},'metric',{cell(length(PRO),length(ALG),obj.app.editA(1).Value)});
                % Initialize the table
                rowName = arrayfun(@class,PRO,'UniformOutput',false);
                for i = length(rowName) : -1 : 2
                    if strcmp(rowName{i},rowName{i-1})
                        rowName{i} = '';
                    end
                end
                obj.app.table.RowName = rowName;
                obj.app.table.Data    = cell(length(PRO),length(ALG));
                obj.TableUpdateColumn();
                % Update the platmetaxGUI
                set([obj.platmetaxGUI.app.button,obj.app.stateA,obj.app.listA,obj.app.dropA,obj.app.editA,obj.app.buttonA],'Enable',false);
                obj.app.listB.Enable      = false;
                obj.app.buttonC(1).Text   = 'Pause';
                obj.app.buttonC(2).Enable = true;
                drawnow('limitrate');
                % Perform the experiment
                for p = 1 : length(PRO)
                    for a = 1 : length(ALG)
                        % Load existing results
                        arrayfun(@(r)obj.ResultLoad(p,a,r),1:size(obj.data.result,3));
                        obj.TableUpdate([],[],p);
                        runIndex = find(reshape(cellfun(@isempty,obj.data.result(p,a,:)),1,[]));
                        %                     20250331修改，去掉勾选Parallel execution的逻辑，防止用户勾选Parallel execution运行报错
%                         if ~isempty(runIndex) && ~isParallel    % Run algorithms in sequence
                        if ~isempty(runIndex)
                            try
                                for r = runIndex
                                    if contains(class(ALG(a).alg), 'Baseoptimizer')
                                        parts = strsplit(class(ALG(a).alg), '_'); % 按下划线分割字符串
                                        comp = parts{1}; % 第一部分
                                        for i = 2:length(parts)-1
                                            comp = strcat(comp, '_', parts{i}); % 重新拼接前两部分
                                        end
                                        env = [comp '_Environment'];
                                        mo = [comp '_Metaoptimizer'];
                                        task = Test(mo, ALG(a).alg, env, PRO(p));
                                        task.run();
                                    elseif contains(class(ALG(a).alg), 'ddpg')
                                        task = Test(@DDPGCMO, ALG(a).alg, @EPSILONCMOAADEnvironment, PRO(p));
                                        task.run();
                                    else
                                        ALG(a).alg.Solve(PRO(p));
                                    end
                                    obj.ResultSave(p,a,r,ALG(a).alg.result,ALG(a).alg.metric);
                                    obj.ResultLoad(p,a,r);
                                    obj.TableUpdate([],[],p);
                                    if strcmp(obj.app.buttonC(2).Enable,'off')
                                        return;
                                    end
                                end
                            catch err
                                uialert(obj.platmetaxGUI.app.figure,'The algorithm terminates unexpectedly, please refer to the command window for details.','Error');
                                obj.cb_stop();
                                rethrow(err);
                            end
%                         elseif ~isempty(runIndex)               % Run algorithms in parallel
%                             try
%                                 Future = arrayfun(@(s)parfeval(@parallelFcn,2,ALG(a).alg,PRO(p)),runIndex);
%                                 while ~all([Future.Read])
%                                     drawnow('limitrate');
%                                     if strcmp(obj.app.buttonC(2).Enable,'off')
%                                         cancel(Future);
%                                         return;
%                                     end
%                                     if strcmp(obj.app.buttonC(1).Text,'Continue')
%                                         waitfor(obj.app.buttonC(1),'Text');
%                                     end
%                                     if strcmp(obj.app.buttonC(2).Enable,'off')
%                                         cancel(Future);
%                                         return;
%                                     end
%                                     [r,result,metric] = fetchNext(Future,0.01);
%                                     if ~isempty(r)
%                                         obj.ResultSave(p,a,runIndex(r),result,metric);
%                                         obj.ResultLoad(p,a,runIndex(r));
%                                         obj.TableUpdate([],[],p);
%                                     end
%                                 end
%                             catch err
%                                 try
%                                     cancel(Future);
%                                 catch
%                                 end
%                                 uialert(obj.platmetaxGUI.app.figure,'The algorithm terminates unexpectedly, please refer to the command window for details.','Error');
%                                 obj.cb_stop();
%                                 rethrow(err);
%                             end
                        end
                    end
                end
                obj.plotConvergence(PRO,ALG)
                obj.cb_stop();
            end
        end
        %% Plot the convergence
        function plotConvergence(obj, pros, algs)
            s  = {'o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
            metric = obj.app.dropC(1).Value;
            if strcmp(metric,'Number of runs') || strcmp(metric,'runtime')
            else
                for p = 1:length(pros)
                    hFig = figure('Visible', 'on'); 
                    hold on;
                    legends = {};
                    for a = 1:length(algs)
                        if isempty(metric)
                            warning('Metric value is empty. Skipping plot.');
                            continue;
                        end
                        convergence = obj.GetConvergenceMetricValue(p, a, metric);
                        if isempty(convergence)
                            warning('Convergence data is empty for algorithm %s. Skipping plot.', class(algs(a).alg));
                            continue;
                        end
                        X=convergence(:,1);
                        meanValues =convergence(:,2);
                        upperBound = meanValues+convergence(:,3);
                        lowerBound = meanValues-convergence(:,3);
                        hMean = plot(X, meanValues, 'LineWidth', 1.5, 'Marker',s{a},'DisplayName', class(algs(a).alg));
                        meanColor = hMean.Color;
                        xfill = [X; flip(X)];
                        yfill = [upperBound; flip(lowerBound)];
                        fill(xfill, yfill, meanColor, 'FaceAlpha', 0.3, 'EdgeColor', 'none','HandleVisibility','off');
                        legendLabel = class(algs(a).alg);
                        if isempty(legendLabel)
                            legendLabel = 'Unknown Algorithm';
                        end
                        legends{end+1} = legendLabel;
                    end
                    legend(legends, 'Location', 'best','Interpreter', 'none');
                    problemClass = class(pros(p));
                    if isempty(problemClass)
                        problemClass = 'Unknown Problem';
                    end
                    xlabel('Number of evaluations');
                    ylabel(metric, 'Interpreter', 'none');
                    title(['Convergence on Problem:', problemClass],'Interpreter', 'none');
                    
                    grid on;
                    hold off;
                end
            end
        end
        
        %% Get the convergence value of metric 
        function score = GetConvergenceMetricValue(obj,p,a,metName)
            metName = strrep(metName,' ','');
            score   = [];
            fes     = [];
            for r = find(reshape(~cellfun(@isempty,obj.data.result(p,a,:)),1,[]))
                if ~isfield(obj.data.metric{p,a,r},metName)
                    obj.data.metric{p,a,r}.(metName) = cellfun(@(S)obj.data.PRO(p).CalMetric(metName,S),obj.data.result{p,a,r}(:,2));
                    metric = obj.data.metric{p,a,r};
                    save(fullfile(obj.data.folder,class(obj.data.ALG(a).alg),sprintf('%s_%s_M%d_D%d_%d.mat',class(obj.data.ALG(a).alg),class(obj.data.PRO(p)),obj.data.PRO(p).M,obj.data.PRO(p).D,r)),'metric','-append');
                end
                try
                    score = [score,obj.data.metric{p,a,r}.(metName)];
                    fes   = [fes,cell2mat(obj.data.result{p,a,r}(:,1))];
                catch
                end
            end
            score = [mean(fes,2),mean(score,2),std(score,0,2)];
        end
        
        %% Stop the execution
        function cb_stop(obj,~,~)
            set([obj.platmetaxGUI.app.button,obj.app.stateA,obj.app.listA,obj.app.dropA,obj.app.editA,obj.app.buttonA],'Enable',true);
            obj.app.listB.Enable      = true;
            obj.app.buttonC(1).Text   = 'Start';
            obj.app.buttonC(2).Enable = false;
        end
        %% Output function
        function outputFcn(obj,Algorithm,Problem)
            assert(strcmp(obj.app.buttonC(2).Enable,'on'),'PlatMetaX:Termination','');
            if strcmp(obj.app.buttonC(1).Text,'Continue')
                waitfor(obj.app.buttonC(1),'Text');
            end
            assert(strcmp(obj.app.buttonC(2).Enable,'on'),'PlatMetaX:Termination','');
        end
        %% Show the specified columns
        function TableUpdateColumn(obj,~,~)
            if ~isempty(obj.data)
                % Update the columns
                str  = get(obj.app.toolC(3:6),'Text');
                show = [obj.app.toolC(3:6).Value];
                obj.app.table.ColumnName   = [str(show)',arrayfun(@(x) class(x.alg), obj.data.ALG, 'UniformOutput', false)];
                obj.app.table.ColumnWidth  = [repmat({45},1,sum(show)),repmat({'auto'},1,length(obj.data.ALG))];
                obj.app.table.ColumnFormat = repmat({'char'},1,sum(show)+length(obj.data.ALG));
                oldsize = size(obj.app.table.Data,2);
                str = arrayfun(@num2str,[obj.data.PRO.N;obj.data.PRO.M;obj.data.PRO.D;obj.data.PRO.maxFE]','UniformOutput',false);
                obj.app.table.Data = [[str(:,show);repmat({''},size(obj.app.table.Data,1)-length(obj.data.PRO),sum(show))],obj.app.table.Data(:,end-length(obj.data.ALG)+1:end)];
                % Update the styles
                styleLoc = cat(1,obj.app.table.StyleConfigurations.TargetIndex{:});
                if ~isempty(styleLoc)
                    obj.app.table.removeStyle();
                    obj.app.table.addStyle(uistyle('FontWeight','bold'),'cell',[styleLoc(:,1),styleLoc(:,2)+size(obj.app.table.Data,2)-oldsize]);
                end
            end
        end
        %% Update the table
        function TableUpdate(obj,~,~,proindex)
            % Change the tooltips of drop-down components
            str = {'Show the mean value','Show the mean value and standard deviation','Show the median value','Show the median value and interquartile range'};
            obj.app.dropC(2).Tooltip = str(obj.app.dropC(2).Value);
            str = {'','Perform the Wilcoxon signed rank test','Perform the Wilcoxon rank sum test','Perform the Friedman''s test'};
            obj.app.dropC(3).Tooltip = str(obj.app.dropC(3).Value);
            str = {'','Highlight the best result','Highlight the best and statistically similar results'};
            obj.app.dropC(4).Tooltip = str(obj.app.dropC(4).Value);
            if ~isempty(obj.data)
                [nP,nA,nR] = size(obj.data.result);
                if nargin < 4
                    proindex = 1 : nP;
                end
                % Delete the cells and styles in the table
                obj.app.table.Data(proindex,end-nA+1:end) = {''};
                styleLoc = cat(1,obj.app.table.StyleConfigurations.TargetIndex{:});
                if ~isempty(styleLoc)
                    styleLoc(ismember(styleLoc(:,1),proindex),:) = [];
                end
                obj.app.table.removeStyle();
                % Identify the metric
                metric = obj.app.dropC(1).Value;
                allMet = [obj.platmetaxGUI.metList;{[0 1]},'Number of runs',{[]};{[1 0]},'runtime',{[]}];
                minMet = allMet{find(ismember(allMet(:,2),metric),1),1}(1,end-1);
                if strcmp(metric,'Number of runs')
                    for p = proindex
                        % Show the number of runs
                        cdata = sum(reshape(~cellfun(@isempty,obj.data.result(p,:,:)),nA,nR),2);
                        obj.app.table.Data(p,end-nA+1:end) = arrayfun(@num2str,cdata','UniformOutput',false);
                    end
                    % Hide the row of statistical results
                    obj.app.table.RowName = obj.app.table.RowName(1:nP);
                    obj.app.table.Data    = obj.app.table.Data(1:nP,:);
                else
                    for p = proindex
                        % Show the metric values
                        valid = find(any(reshape(~cellfun(@isempty,obj.data.result(p,:,:)),nA,nR),2))';
                        cdata = cell(1,nA);     % All metric values
                        mdata = zeros(1,nA);    % Mean or median value
                        sdata = zeros(1,nA);    % STD or IQR value
                        for a = valid
                            if nargin < 4
                                drawnow('limitrate');
                            end
                            cdata{a} = obj.GetMetricValue(p,a,metric,false);
                            datapure = cdata{a}(~isnan(cdata{a}));
                            if obj.app.dropC(2).Value < 3
                                mdata(a) = mean(datapure);
                                sdata(a) = std(datapure);
                            else
                                mdata(a) = median(datapure);
                                sdata(a) = iqr(datapure);
                            end
                            if ismember(obj.app.dropC(2).Value,[1 3])
                                str = sprintf('%.4e',mdata(a));
                            else
                                str = sprintf('%.4e (%.2e)',mdata(a),sdata(a));
                            end
                            obj.app.table.Data{p,end-nA+a} = strrep(strrep(str,'e-0','e-'),'e+0','e+');
                        end
                        % Highlight the best metric value
                        valid(arrayfun(@(s)isnan(s),mdata(valid))) = [];
                        if ~isempty(valid) && obj.app.dropC(4).Value > 1
                            if minMet
                                [~,best] = min(mdata(valid));
                            else
                                [~,best] = max(mdata(valid));
                            end
                            styleLoc = [styleLoc;p,size(obj.app.table.Data,2)-nA+valid(best)];
                        end
                        % Calculate the statistical test results
                        if obj.app.dropC(3).Value > 1 && length(valid) > 1 && ismember(nA,valid)
                            minlen = min(cellfun(@length,cdata(valid)));
                            if minlen > 1
                                vdata = cellfun(@(s)s(1:minlen),cdata(valid),'UniformOutput',false);
                                if obj.app.dropC(3).Value == 4
                                    [~,~,stats] = friedman(cell2mat(vdata),1,'off');
                                    c     = multcompare(stats,'Display','off');
                                    diff1 = c(any(c==length(vdata),2),end) < 0.05;
                                    if obj.app.dropC(4).Value == 3
                                        diff2 = c(any(c==best,2),end) < 0.05;
                                    end
                                elseif obj.app.dropC(3).Value == 3
                                    diff1 = cellfun(@(s)ranksum(s,vdata{end})<0.05,vdata(:,1:end-1));
                                    if obj.app.dropC(4).Value == 3
                                        diff2 = cellfun(@(s)ranksum(s,vdata{best})<0.05,vdata(:,[1:best-1,best+1:end]));
                                    end
                                elseif obj.app.dropC(3).Value == 2
                                    diff1 = cellfun(@(s)signrank(s,vdata{end})<0.05,vdata(:,1:end-1));
                                    if obj.app.dropC(4).Value == 3
                                        diff2 = cellfun(@(s)signrank(s,vdata{best})<0.05,vdata(:,[1:best-1,best+1:end]));
                                    end
                                end
                                for a = 1 : length(diff1)
                                    if ~diff1(a) || mdata(valid(a))==mdata(nA)
                                        obj.app.table.Data{p,end-nA+valid(a)} = [obj.app.table.Data{p,end-nA+valid(a)},' ='];
                                    elseif mdata(valid(a))<mdata(nA)&&minMet || mdata(valid(a))>mdata(nA)&&~minMet
                                        obj.app.table.Data{p,end-nA+valid(a)} = [obj.app.table.Data{p,end-nA+valid(a)},' +'];
                                    else
                                        obj.app.table.Data{p,end-nA+valid(a)} = [obj.app.table.Data{p,end-nA+valid(a)},' -'];
                                    end
                                    if obj.app.dropC(4).Value == 3 && ~diff2(a)
                                        styleLoc = [styleLoc;p,size(obj.app.table.Data,2)-nA+valid(a)+(a>=best)];
                                    end
                                end
                            end
                        end
                    end
                    if ~isempty(styleLoc)
                        obj.app.table.addStyle(uistyle('FontWeight','bold'),'cell',styleLoc);
                    end
                    % Count the statistical test results
                    if obj.app.dropC(3).Value > 1
                        if length(obj.app.table.RowName) == nP
                            obj.app.table.RowName = [obj.app.table.RowName;'+/-/='];
                            obj.app.table.Data    = [obj.app.table.Data;repmat({''},1,size(obj.app.table.Data,2))];
                        end
                        sign1 = cellfun(@(s)~isempty(s)&&strcmp('+',s(end)),obj.app.table.Data(1:end-1,end-nA+1:end));
                        sign2 = cellfun(@(s)~isempty(s)&&strcmp('-',s(end)),obj.app.table.Data(1:end-1,end-nA+1:end));
                        sign3 = cellfun(@(s)~isempty(s)&&strcmp('=',s(end)),obj.app.table.Data(1:end-1,end-nA+1:end));
                        for a = 1 : nA-1
                        	obj.app.table.Data{end,end-nA+a} = sprintf('%d/%d/%d',sum(sign1(:,a)),sum(sign2(:,a)),sum(sign3(:,a)));
                        end
                    else
                        obj.app.table.RowName = obj.app.table.RowName(1:nP);
                        obj.app.table.Data    = obj.app.table.Data(1:nP,:);
                    end
                end
            end
        end
        %% Load the result file
        function ResultLoad(obj,p,a,r)
            try
                filename = fullfile(obj.data.folder,class(obj.data.ALG(a).alg),sprintf('%s_%s_M%d_D%d_%d.mat',class(obj.data.ALG(a).alg),class(obj.data.PRO(p)),obj.data.PRO(p).M,obj.data.PRO(p).D,r));
                load(filename,'-mat','result','metric');
                obj.data.result{p,a,r} = result;
                obj.data.metric{p,a,r} = metric;
            catch
            end
        end
        %% Save the result file
        function ResultSave(obj,p,a,r,result,metric)
            folder   = fullfile(obj.data.folder,class(obj.data.ALG(a).alg));
            [~,~]    = mkdir(folder);
            filename = fullfile(folder,sprintf('%s_%s_M%d_D%d_%d.mat',class(obj.data.ALG(a).alg),class(obj.data.PRO(p)),obj.data.PRO(p).M,obj.data.PRO(p).D,r));
            save(filename,'result','metric');
        end
        %% Get the metric value 
        function score = GetMetricValue(obj,p,a,metName,showAll)
            metName = strrep(metName,' ','');
            score   = [];
            fes     = [];
            for r = find(reshape(~cellfun(@isempty,obj.data.result(p,a,:)),1,[]))
                if ~isfield(obj.data.metric{p,a,r},metName)
                    obj.data.metric{p,a,r}.(metName) = cellfun(@(S)obj.data.PRO(p).CalMetric(metName,S),obj.data.result{p,a,r}(:,2));
                    metric = obj.data.metric{p,a,r};
                    save(fullfile(obj.data.folder,class(obj.data.ALG(a).alg),sprintf('%s_%s_M%d_D%d_%d.mat',class(obj.data.ALG(a).alg),class(obj.data.PRO(p)),obj.data.PRO(p).M,obj.data.PRO(p).D,r)),'metric','-append');
                end
                try
                    if showAll  % Mean convergence profile of the metric
                        score = [score,obj.data.metric{p,a,r}.(metName)];
                        fes   = [fes,cell2mat(obj.data.result{p,a,r}(:,1))];
                    else        % All metric values of the last populations
                        score = [score;obj.data.metric{p,a,r}.(metName)(end)];
                    end
                catch
                end
            end
            if showAll
                score = [mean(fes,2),mean(score,2)];
            end
        end
        %% Save the table
        function cb_save(obj,~,~)
            if ~isempty(obj.app.table.Data)
                try
                    [Name,Path] = uiputfile({'*.xlsx','Excel table';'*.tex','TeX table';'*.txt','Text file';'*.mat','MAT file'},'','new');
                    figure(obj.platmetaxGUI.app.figure);
                    if ischar(Name)
                        [~,~,Type] = fileparts(Name);
                        switch Type
                            case '.xlsx'
                                table2excel(fullfile(Path,Name),obj.app.dropC(1).Value,[{'Problem'},obj.app.table.ColumnName';obj.app.table.RowName,obj.app.table.Data],cat(1,obj.app.table.StyleConfigurations.TargetIndex{:}),size(obj.data.result,2));
                            case '.tex'
                                table2tex(fullfile(Path,Name),[{'Problem'},obj.app.table.ColumnName';obj.app.table.RowName,obj.app.table.Data],cat(1,obj.app.table.StyleConfigurations.TargetIndex{:}),size(obj.data.result,1),size(obj.data.result,2));
                            case '.txt'
                                table2txt(fullfile(Path,Name),[{'Problem'},obj.app.table.ColumnName';obj.app.table.RowName,obj.app.table.Data]);
                        	case '.mat'
                                table2mat(fullfile(Path,Name),obj.data.metric,obj.app.dropC(1).Value);
                        end
                    end
                catch err
                    uialert(obj.platmetaxGUI.app.figure,'Fail to save the table, please refer to the command window for details.','Error');
                    rethrow(err);
                end
            end
        end
        %% Select the cells of table
        function cb_tableSelect(obj,~,event)
            if ~isempty(obj.data)
                grids = [event.Indices(:,1),event.Indices(:,2)-size(obj.app.table.Data,2)+size(obj.data.result,2)];
                grids(grids(:,1)>size(obj.data.result,1)|grids(:,2)<1,:) = [];
                if ~isempty(grids) && ~ismember(obj.app.dropC(1).Value,{'Number of runs','runtime'})
                    obj.app.table.UserData = [min(grids,[],1),max(grids,[],1)];
                else
                    obj.app.table.UserData = [];
                end
            end
        end
        %% Show the menu of result display
        function cb_tableDisplay(obj,~,~)
            if ~isempty(obj.app.table.UserData)
                obj.app.tablemenu.show();
            else
                uialert(obj.platmetaxGUI.app.figure,'Please select at least one cell of metric value in the table.','Error');
            end
        end
        %% Show the results in new figure
        function cb_tableShow(obj,~,~,type)
            metric = obj.app.dropC(1).Value;
            loc    = obj.app.table.UserData;
            nRow   = loc(3) - loc(1) + 1;
            nCol   = loc(4) - loc(2) + 1;
            if type < 3
                movegui(figure('NumberTitle','off','Name','','Position',[0 0 240*nCol 220*nRow]),'center');
                for r = 1 : nRow
                    for c = 1 : nCol
                        ax    = Draw(axes('Unit','pixels','Position',[(c-1)*240+35 (nRow-r)*220+35 200 170]));
                        valid = find(reshape(~cellfun(@isempty,obj.data.result(r+loc(1)-1,c+loc(2)-1,:)),1,[]));
                        if ~isempty(valid)
                            [~,rank] = sort(obj.GetMetricValue(r+loc(1)-1,c+loc(2)-1,metric,false));
                            if type == 1
                                obj.data.PRO(r+loc(1)-1).DrawObj(obj.data.result{r+loc(1)-1,c+loc(2)-1,valid(rank(ceil(end/2)))}{end});
                            elseif type == 2
                                obj.data.PRO(r+loc(1)-1).DrawDec(obj.data.result{r+loc(1)-1,c+loc(2)-1,valid(rank(ceil(end/2)))}{end});
                            end
                        end
                        set(ax,'FontSize',8);
                        set(ax.Children,{'MarkerSize','LineWidth'},{4,0.6});
                        title([class(obj.data.ALG(c+loc(2)-1)),' on ',class(obj.data.PRO(r+loc(1)-1))],'Interpreter','none');
                    end
                end
            else
                movegui(figure('NumberTitle','off','Name','','Position',[0 0 290*nRow 260]),'center');
                for r = 1 : nRow
                    ax = Draw(axes('Unit','pixels','Position',[(r-1)*290+50 40 220 200]));
                    s  = {'o','+','s','*','^','x';'-k','--k','-b','--b','-g','--g'};
                    for c = 1 : nCol
                        value = obj.GetMetricValue(r+loc(1)-1,c+loc(2)-1,metric,true);
                        Draw(value,[s{1,mod(c-1,size(s,2))+1},s{2,mod(ceil(c/size(s,2))-1,size(s,2))+1}],'MarkerSize',5,'LineWidth',0.6);
                    end
                    legend(ax,arrayfun(@(s)class(s),obj.data.ALG(loc(2):loc(4)),'UniformOutput',false),'Location','best');
                    set(ax,'FontSize',10);
                    [ax.XLabel.String,ax.YLabel.String,ax.Title.String,ax.Title.Interpreter] = deal('Number of function evaluations',strrep(metric,'_',' '),class(obj.data.PRO(r+loc(1)-1)),'none');
                end
            end
        end
    end
end

%% Function for parallelization
function [result,metric] = parallelFcn(Algorithm,Problem)
    Algorithm.Solve(Problem);
    result = Algorithm.result;
    metric = Algorithm.metric;
end

%% Save the table to Excel
function table2excel(filename,sheetname,Data,styleLoc,nA)
    [x,y] = size(Data);
    % Convert the indices to Excel cell number
    function range = getRange(varargin)
        if nargin == 2
            range = num2str(varargin{1});
            while varargin{2} > 0
                range = [char(65+mod(varargin{2}-1,26)),range];
                varargin{2} = floor((varargin{2}-1)/26);
            end
        else
            range = [getRange(varargin{1:2}),':',getRange(varargin{3:4})];
        end
    end
    % Open the file and get the sheet
    try
        Excel = actxGetRunningServer('Excel.Application');
    catch
        Excel = actxserver('Excel.Application');
    end
    if exist(filename,'file')
        delete(filename);
    end
    Workbook = invoke(Excel.Workbooks,'Add');
    Workbook.SaveAs(filename);
    Sheet = Workbook.ActiveSheet;
    Sheet.Name = sheetname;
    % Set the column width
    head = y - nA;
    Sheet.Range(getRange(1,1,x,1)).ColumnWidth = 10;
    if head >= 2
        Sheet.Range(getRange(1,2,x,head)).ColumnWidth = 6;
    end
    Sheet.Range(getRange(1,head+1,x,y)).ColumnWidth = 22;
    % Initialize the cells
    Range = Sheet.Range(getRange(1,1,x,y));
    Range.HorizontalAlignment = 3;
    Range.Font.Name = 'Times New Roman';
    if strcmp(Data{x,1},'+/-/=')
        Sheet.Range(getRange(x,1,x,head)).Merge;
        Sheet.Range(getRange(x,1,x,y)).NumberFormat = '@';
    end
    % Set the font color
    for i = 1 : size(styleLoc,1)
        Sheet.Range(getRange(styleLoc(i,1)+1,styleLoc(i,2)+1)).Font.Color = 15282995;
    end
    % Write the data
    Range.Value = Data;
    % Set the border and merge the cells
    Range.Borders.LineStyle = 1;
    for i = 2 : x
        if isempty(Data{i,1})
            Sheet.Rows.Item(i-1).Borders.Item(4).Linestyle = 0;
            Sheet.Rows.Item(i).Borders.Item(3).Linestyle   = 0;
            Sheet.Range(getRange(i-1,1,i,1)).Merge;
        end
    end
    % Close the file
    Workbook.Save;
    Workbook.Close;
    Excel.Quit;
    Excel.delete;
end

%% Save the table to TeX
function table2tex(filename,Data,styleLoc,nP,nA)
    % Convert the data
    mainData = Data(2:nP+1,end-nA+1:end);
    mainData = regexprep(mainData,'+$','$+$');
    mainData = regexprep(mainData,'-$','$-$');
    mainData = regexprep(mainData,'=$','$\\approx$');
    for i = 1 : size(styleLoc,1)
        mainData{styleLoc(i,1),styleLoc(i,2)-(size(Data,2)-nA-1)} = ['\hl{',mainData{styleLoc(i,1),styleLoc(i,2)-(size(Data,2)-nA-1)},'}'];
    end
    Data(2:nP+1,end-nA+1:end) = mainData;
    Data(end,1)          = regexprep(Data(end,1),'^\+/\-/=$',['\\multicolumn{',num2str(size(Data,2)-nA),'}{c}{$+/-/\\approx$}']);
    Data(1,2:end-nA)     = strcat('$',Data(1,2:end-nA),'$');
    Data(1,end-nA+1:end) = regexprep(Data(1,end-nA+1:end),'_','\\_');
    noEmpty = ~cellfun(@isempty,Data(:,1));
    for i = 2 : nP+1
        if noEmpty(i)
            Data{i,1} = sprintf('\\multirow{%d}{*}{%s}',find([noEmpty(i+1:end);true],1),Data{i,1});
        end
    end
    % Generate the TeX code
    Code = eval(sprintf('strcat(%s)',strjoin(arrayfun(@(S)num2str(S,'Data(:,%d)'),1:size(Data,2),'UniformOutput',false),',''&'',')));
    Code = strcat(Code,'\\');
    if ~isempty(regexp(Code{end,1},'^\\multicolumn','once'))
        temp = strfind(Code{end,1},'&');
        Code{end,1}(temp(1:size(Data,2)-nA-1)) = [];
    end
    noEmpty = find(noEmpty);
    for i = 3 : length(noEmpty)
        Code = [Code(1:noEmpty(i)+i-4);'\hline';Code(noEmpty(i)+i-3:end)];
    end
    Code = ['\documentclass[journal]{IEEEtran}'
            '\usepackage{multirow,booktabs,color,soul,threeparttable}'
            '\definecolor{hl}{rgb}{0.75,0.75,0.75}'
            '\sethlcolor{hl}'
            '\begin{document}'
            '\begin{table*}[htbp]'
            '\renewcommand{\arraystretch}{1.2}'
            '\centering'
            '\caption{No Title}'
            ['\begin{tabular}{',repmat('c',1,size(Data,2)),'}']
            '\toprule'
            Code(1)
            '\midrule'
            Code(2:end)
            '\bottomrule'
            '\end{tabular}'
            '\label{No Label}'
            '\end{table*}'
            '\end{document}'];
    fid = fopen(filename,'wt');
    for i = 1 : length(Code)
        fprintf(fid,'%s\n',Code{i});
    end
    fclose(fid);
end

%% Save the table to .txt
function table2txt(filename,Data)
    fid = fopen(filename,'wt');
    for i = 1 : size(Data,1)
        fprintf(fid,'%s\n',strjoin(Data(i,:),'\t'));
    end
    fclose(fid);
end

%% Save the table to .mat
function table2mat(filename,Metric,metricName)
    metricName = strrep(metricName,' ','');
    if strcmp(metricName,'Numberofruns')
        Data = sum(~cellfun(@isempty,Metric),3);
    else
        Data = nan(size(Metric));
        for i = 1 : numel(Metric)
            if isfield(Metric{i},metricName)
                Data(i) = Metric{i}.(metricName)(end);
            end
        end
    end
    eval([metricName,'=Data;'])
    save(filename,metricName);
end