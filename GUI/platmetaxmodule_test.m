classdef platmetaxmodule_test < handle
%module_test - Test module.

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
        app  = struct(); 	% All the components
        data = {};          % All the results
    end
    methods(Access = ?platmetaxGUI)
        %% Constructor
        function obj = platmetaxmodule_test(platmetaxGUI)
            % The main grid
            obj.platmetaxGUI = platmetaxGUI;
            obj.app.maingrid = platmetaxGUI.APP(3,1,uigridlayout(obj.platmetaxGUI.app.maingrid,'RowHeight',{'1x','0.3x'},'ColumnWidth',{'0.9x',1,40,'1x',5},'Padding',[0 5 0 5],'RowSpacing',5,'ColumnSpacing',0,'BackgroundColor','w'));
            
            % The first panel
            obj.app.gridNew(1)   = platmetaxGUI.APP(1,1,uigridlayout(obj.app.maingrid,'RowHeight',{18,'1x'},'ColumnWidth',{'1x','1x','1x','1x','2x','1x'},'BackgroundColor','w'));
            obj.app.label(3) = platmetaxGUI.APP(1,[2 5],uilabel(obj.app.gridNew(1),'Text','Result display','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.dropC(1) = platmetaxGUI.APP(1,5,uidropdown(obj.app.gridNew(1),'BackgroundColor','w','Items',{},'ValueChangedFcn',@obj.cb_slider));
            obj.app.dropC(2) = platmetaxGUI.APP(1,5,uidropdown(obj.app.gridNew(1),'BackgroundColor','w','Items',{},'ValueChangedFcn',@obj.cb_slider,'Visible','off'));
            obj.app.grid(3)  = platmetaxGUI.APP(2,[1 6],uigridlayout(obj.app.gridNew(1),'RowHeight',{'1x',40,30},'ColumnWidth',{20,150,'1x','1x',120,30,20},'Padding',[15 10 15 0],'RowSpacing',5,'BackgroundColor','w'));
            obj.app.axes     = platmetaxGUI.APP(1,[1 6],uiaxes(obj.app.grid(3),'BackgroundColor','w','Box','on'));
            obj.app.waittip  = platmetaxGUI.APP(1,[1 6],uilabel(obj.app.grid(3),'HorizontalAlignment','center','Text','                 Please wait ... ...','Visible','off'));
            tempTb = axtoolbar(obj.app.axes(1),{'rotate','pan','zoomin','zoomout'});
            obj.app.toolC(1)   = axtoolbarbtn(tempTb,'push','Icon',obj.platmetaxGUI.icon.gif,'Tooltip','Save the evolutionary process to gif','ButtonPushedFcn',@obj.cb_toolbutton1);
            obj.app.toolC(2)   = axtoolbarbtn(tempTb,'push','Icon',obj.platmetaxGUI.icon.newfigure,'Tooltip','Open in new figure and save to workspace','ButtonPushedFcn',@obj.cb_toolbutton2);
            obj.app.slider     = platmetaxGUI.APP(2,[1 6],uislider(obj.app.grid(3),'Limits',[0 1],'MajorTicks',0:0.25:1,'MajorTickLabels',{'0%','25%','50%','75%','100%'},'MinorTicks',0:0.01:1,'ValueChangedFcn',@obj.cb_slider));
            obj.app.labelC     = platmetaxGUI.APP(3,[1 2],uilabel(obj.app.grid(3),'Text','','Fontcolor',[0.0,0.0,0.0],'HorizontalAlignment','left'));

            platmetaxGUI.APP([1 2],2,uipanel(obj.app.maingrid,'BackgroundColor',[.8 .8 .8]));
            % The second panel
            obj.app.grid(1)   = platmetaxGUI.APP(1,4,uigridlayout(obj.app.maingrid,'RowHeight',{18,16,21,16,21,16,21,20,'1x'},'ColumnWidth',{'1x'},'Padding',[0 10 0 0],'RowSpacing',3,'ColumnSpacing',5,'BackgroundColor','w'));
            obj.app.label(1) = platmetaxGUI.APP(1,[1 9],uilabel(obj.app.grid(1),'Text','Algorithm selection','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            [obj.app.stateA,obj.app.labelA] = platmetaxGUI.GenerateLabelButtonTest(obj.app.grid(1),[0,1,0,1,zeros(1,13)],@obj.cb_filter);
            obj.app.labelA(4) = platmetaxGUI.APP(8,[1 2],uilabel(obj.app.grid(1),'Text','Algorithms','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.labelA(5) = platmetaxGUI.APP(8,4,uilabel(obj.app.grid(1),'HorizontalAlignment','right','FontSize',10,'FontColor',[0.93,0.69,0.13]));
            obj.app.listA(1)  = platmetaxGUI.APP(9,[1 4],uilistbox(obj.app.grid(1),'FontColor',[0.93,0.69,0.13]));
            obj.app.labelA(6) = platmetaxGUI.APP(8,[6 7],uilabel(obj.app.grid(1),'Text','Problems','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.labelA(7) = platmetaxGUI.APP(8,9,uilabel(obj.app.grid(1),'HorizontalAlignment','right','FontSize',10,'FontColor',[0.30,0.75,0.93]));
            obj.app.listA(2)  = platmetaxGUI.APP(9,[6 9],uilistbox(obj.app.grid(1),'FontColor',[0.30,0.75,0.93]));
            obj.app.dropA(1)  = platmetaxGUI.APP(8,[2 3],uidropdown(obj.app.grid(1),'BackgroundColor','w','FontColor',[0.93,0.69,0.13],'Items',{'All year'},'ValueChangedFcn',@(h,~)platmetaxGUI.UpdateAlgProListYear(obj.app.listA(1),h,obj.app.labelA(5),obj.platmetaxGUI.algList)));
            obj.app.dropA(2)  = platmetaxGUI.APP(8,[7 8],uidropdown(obj.app.grid(1),'BackgroundColor','w','FontColor',[0.30,0.75,0.93],'Items',{'All year'},'ValueChangedFcn',@(h,~)platmetaxGUI.UpdateAlgProListYear(obj.app.listA(2),h,obj.app.labelA(7),obj.platmetaxGUI.proList)));

            
            % The third panel
            obj.app.grid(4)  = platmetaxGUI.APP(2,1,uigridlayout(obj.app.maingrid,'RowHeight',{18,22,'1x'},'ColumnWidth',{'1.2x','1x','1x'},'Padding',[40 0 60 0],'RowSpacing',15,'BackgroundColor','w'));
            obj.app.label(4) = platmetaxGUI.APP(1,[1 4],uilabel(obj.app.grid(4),'Text','Result selection','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.dropD(1) = platmetaxGUI.APP(2,1,uidropdown(obj.app.grid(4),'BackgroundColor','w','Items',{},'ItemsData',1:999,'ValueChangedFcn',@obj.cb_dropdown1));
            obj.app.dropD(2) = platmetaxGUI.APP(2,2,uidropdown(obj.app.grid(4),'BackgroundColor','w','Items',{},'ValueChangedFcn',@obj.cb_dropdown2));
            obj.app.dropD(3) = platmetaxGUI.APP(2,2,uidropdown(obj.app.grid(4),'BackgroundColor','w','Items',{},'ValueChangedFcn',@obj.cb_dropdown2,'Visible','off'));
            obj.app.labelD   = platmetaxGUI.APP(2,3,uilabel(obj.app.grid(4),'Text','','HorizontalAlignment','right'));
            obj.app.textD    = platmetaxGUI.APP(3,[1 4],uitextarea(obj.app.grid(4),'Editable','off'));
           
            % The fourth panel
            obj.app.gridNew(2)  = platmetaxGUI.APP(2,4,uigridlayout(obj.app.maingrid,'RowHeight',{18,'1x',30},'ColumnWidth',{'1x','1x'},'Padding',[12 10 12 0],'RowSpacing',15,'BackgroundColor','w'));       
            obj.app.label(2) = platmetaxGUI.APP(1,[1 4],uilabel(obj.app.gridNew(2),'Text','Parameter setting','HorizontalAlignment','center','FontSize',13,'FontColor',[0.00,0.00,0.00],'FontWeight','bold'));
            obj.app.gridNew(3)  = platmetaxGUI.APP(2,[1 2],uigridlayout(obj.app.gridNew(2),'RowHeight',{'1x'},'ColumnWidth',{'1x'},'Padding',[12 10 12 0],'RowSpacing',15,'BackgroundColor','w'));
            obj.app.gridNew(4)  = platmetaxGUI.APP(2,[3 4],uigridlayout(obj.app.gridNew(2),'RowHeight',{'1x'},'ColumnWidth',{'1x'},'Padding',[12 10 12 0],'RowSpacing',15,'BackgroundColor','w'));
            obj.app.listB(1)   = uilist(obj.app.gridNew(3),obj.platmetaxGUI.app.figure,obj.platmetaxGUI.icon);
            obj.app.listB(2)   = uilist(obj.app.gridNew(4),obj.platmetaxGUI.app.figure,obj.platmetaxGUI.icon);
            obj.app.listA(1).ValueChangedFcn = @(~,~)platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(1),obj.app.listB(1),'BASEOPTIMIZER',1);
            obj.app.listA(2).ValueChangedFcn = @(~,~)platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(2),obj.app.listB(2),'PROBLEM',-1);
            obj.app.buttonC(1) = platmetaxGUI.APP(3,1,uibutton(obj.app.gridNew(2),'push','Text','Start','FontColor',[1 1 1],'FontSize',16,"BackgroundColor",[0.07,0.62,1.00],'ButtonpushedFcn',@obj.cb_start));
            obj.app.buttonC(2) = platmetaxGUI.APP(3,2,uibutton(obj.app.gridNew(2),'push','Text','Stop','FontColor',[1 1 1],'FontSize',16,'Enable','off',"BackgroundColor",[0.07,0.62,1.00],'ButtonpushedFcn',@(~,~)set(obj.app.buttonC(1:2),{'Enable','Text'},{true,'Start';false,'Stop'})));
            obj.app.menuC      = uicontext(obj.platmetaxGUI.app.figure,120);
            obj.app.menuC.add('  Save best solutions','',{@obj.cb_save,1});
            obj.app.menuC.add('  Save all solutions','',{@obj.cb_save,2});
            obj.app.menuC.flush();
            obj.app.buttonC(3) = platmetaxGUI.APP(3,4,uibutton(obj.app.gridNew(2),'push','Text','Save','FontColor',[1 1 1],'FontSize',16,'Enable','off',"BackgroundColor",[0.07,0.62,1.00],'ButtonpushedFcn',@(~,~)obj.app.menuC.show()));
            
            % Initialization
            obj.cb_filter([],[],2);
            obj.app.listA(1).Value = 'NSGAII';
            obj.app.listA(2).Value = 'DTLZ2';
            platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(1),obj.app.listB(1),'BASEOPTIMIZER',1);
            platmetaxGUI.UpdateAlgProPara(obj.platmetaxGUI.app.figure,obj.app.listA(2),obj.app.listB(2),'PROBLEM',-1);
        end
    end
    methods(Access = private)
        %% Update the algorithms and problems in the lists
        function cb_filter(obj,~,~,index)
            % Update the lists of algorithms and problems
            func = platmetaxGUI.UpdateAlgProList(index,obj.app.stateA,obj.app.listA(1),obj.app.dropA(1),obj.app.labelA(5),obj.platmetaxGUI.algList,obj.app.listA(2),obj.app.dropA(2),obj.app.labelA(7),obj.platmetaxGUI.proList);
            % Update the list of metrics
            show = cellfun(@(s)func(s(2:end,1:end-2)),obj.platmetaxGUI.metList(:,1));
            if obj.app.stateA(1).Value == 0 % Multi-objective optimization
                obj.app.dropC(1).Items = ['Population (objectives)';'Population (variables)';'True Pareto front';obj.platmetaxGUI.metList(show,2)];
                obj.app.dropD(2).Items = ['runtime';obj.platmetaxGUI.metList(show,2)];
            else                            % Single-objective optimization
                obj.app.dropC(2).Items = ['Population (variables)';obj.platmetaxGUI.metList(show,2)];
                obj.app.dropD(3).Items = ['runtime';obj.platmetaxGUI.metList(show,2)];
            end
            obj.cb_dropdown2();
        end
        %% Start the execution
        function cb_start(obj,~,~)
            if strcmp(obj.app.buttonC(1).Text,'Pause')
                obj.app.buttonC(1).Text = 'Continue';
            elseif strcmp(obj.app.buttonC(1).Text,'Continue')
                obj.app.buttonC(1).Text = 'Pause';
            else
                % Generate the ALGORITHM and PROBLEM objects
                try
                    [name,para] = platmetaxGUI.GetParameterSetting(obj.app.listB(1).items);
                    ALG = feval(name,'parameter',para,'outputFcn',@obj.outputFcn,'save',20);
                    [name,para] = platmetaxGUI.GetParameterSetting(obj.app.listB(2).items);
                    PRO = feval(name,'N',para{1},'M',para{2},'D',para{3},obj.app.listB(2).items.label(4).Text,para{4},'parameter',para(5:end));
                catch err
                    uialert(obj.platmetaxGUI.app.figure,err.message,'Invalid parameter settings');
                    return;
                end
                % Update the data
                str = sprintf('<Algorithm: %s>\n',class(ALG));
                for i = 1 : length(obj.app.listB(1).items.label)
                    str = [str,sprintf('%s: %s\n',obj.app.listB(1).items.label(i).Text,obj.app.listB(1).items.edit(i).Value)];
                end
                str = [str,sprintf('\n<Problem: %s>\n',class(PRO))];
                for i = 1 : length(obj.app.listB(2).items.label)
                    str = [str,sprintf('%s: %s\n',obj.app.listB(2).items.label(i).Text,obj.app.listB(2).items.edit(i).Value)];
                end
                obj.data = [obj.data;{ALG},{PRO},{str}];
                % Update the platmetaxGUI
                set([obj.platmetaxGUI.app.button,obj.app.stateA,obj.app.listA,obj.app.dropA,obj.app.dropD,obj.app.dropC],'Enable',false);
                obj.app.listB(1).Enable = false;
                obj.app.listB(2).Enable = false;
                set(obj.app.toolC,'Visible',false);
                obj.app.buttonC(1).Text = 'Pause';
                set(obj.app.buttonC([2,3]),'Enable',true);
                if PRO.M > 1
                    obj.app.dropC(1).Value = obj.app.dropC(1).Items{1};
                    set([obj.app.dropC(1),obj.app.dropD(2)],'Visible',true);
                    set([obj.app.dropC(2),obj.app.dropD(3)],'Visible',false);
                else
                    obj.app.dropC(2).Value = obj.app.dropC(2).Items{1};
                    set([obj.app.dropC(1),obj.app.dropD(2)],'Visible',false);
                    set([obj.app.dropC(2),obj.app.dropD(3)],'Visible',true);
                end
                obj.app.dropD(1).Items = [obj.app.dropD(1).Items,sprintf('%s on %s',class(ALG),class(PRO))];
                obj.app.dropD(1).Value = length(obj.app.dropD(1).Items);
                obj.app.labelD.Text    = '';
                obj.app.textD.Value    = '';
                % Execute the algorithm
                try
                    algName = class(ALG);
                    if contains(algName, 'Baseoptimizer')
                        parts = strsplit(algName, '_'); % 按下划线分割字符串
                        comp = parts{1}; % 第一部分
                        for i = 2:length(parts)-1
                            comp = strcat(comp, '_', parts{i}); % 重新拼接前两部分
                        end
                        env = str2func([comp '_Environment']);
                        mo = str2func([comp '_Metaoptimizer']);
                        task = Test(mo, ALG, env, PRO);
                        task.run();
                        obj.cb_stop();
                    else
                        ALG.Solve(PRO);
                        obj.cb_stop();
                    end
                catch err
                    uialert(obj.platmetaxGUI.app.figure,'The algorithm terminates unexpectedly, please refer to the command window for details.','Error');
                    obj.cb_stop();
                    rethrow(err);
                end
            end
        end
        %% Stop the execution
        function cb_stop(obj,~,~)
            set([obj.platmetaxGUI.app.button,obj.app.stateA,obj.app.listA,obj.app.dropA,obj.app.dropD,obj.app.dropC],'Enable',true);
            obj.app.listB(1).Enable = true;
            obj.app.listB(2).Enable = true;
            set(obj.app.toolC,'Visible',true);
            obj.app.buttonC(1).Text   = 'Start';
            obj.app.buttonC(2).Enable = false;
            if isempty(obj.data{end,1}.result)
                obj.data(end,:)             = [];
                obj.app.dropD(1).Items(end) = [];
            end
            obj.cb_dropdown1();
        end
        %% Save the result
        function cb_save(obj,~,~,type)
            ALG   = obj.data{obj.app.dropD(1).Value,1};
            PRO   = obj.data{obj.app.dropD(1).Value,2};
            rate  = PRO.FE/max(PRO.FE,PRO.maxFE);
            index = max(1,round(obj.app.slider.Value/rate*size(ALG.result,1)));
            platmetaxGUI.SavePopulation(obj.platmetaxGUI.app.figure,ALG.result{index,2},type);
        end
        %% Output function
        function outputFcn(obj,Algorithm,Problem)
            obj.app.slider.Value = Problem.FE/max(Problem.FE,Problem.maxFE);
            obj.cb_slider();
            assert(strcmp(obj.app.buttonC(2).Enable,'on'),'PlatMetaX:Termination','');
            if strcmp(obj.app.buttonC(1).Text,'Continue')
                waitfor(obj.app.buttonC(1),'Text');
            end
            assert(strcmp(obj.app.buttonC(2).Enable,'on'),'PlatMetaX:Termination','');
        end
        %% Show the specified data
        function cb_slider(obj,~,~,ax)
            if ~isempty(obj.app.dropD(1).Items)
                % Determine the current number of evaluationsnumber of evaluations
                ALG  = obj.data{obj.app.dropD(1).Value,1};
                PRO  = obj.data{obj.app.dropD(1).Value,2};
                rate = PRO.FE/max(PRO.FE,PRO.maxFE);
                obj.app.slider.Value      = min(obj.app.slider.Value,rate);
                obj.app.slider.MajorTicks = 0 : 0.25 : rate;
                obj.app.slider.MinorTicks = 0 : 0.01 : rate;
                index = max(1,round(obj.app.slider.Value/rate*size(ALG.result,1)));
                obj.app.labelC.Text = sprintf('%d evaluations',ALG.result{index,1});
                % Clear the default or specified axes
                if nargin > 3
                    Draw(ax);
                else
                    Draw(obj.app.axes);
                end
                isMetric = false;
                if PRO.M > 1    % Multi-objective optimization
                    switch obj.app.dropC(1).Value
                        case 'Population (objectives)'
                            PRO.DrawObj(ALG.result{index,2});
                        case 'Population (variables)'
                            PRO.DrawDec(ALG.result{index,2});
                        case 'True Pareto front'
                            Draw(PRO.optimum,{'\it f\rm_1','\it f\rm_2','\it f\rm_3'});
                        otherwise
                            obj.app.waittip.Visible = 'on'; drawnow();
                            Draw([cell2mat(ALG.result(:,1)),ALG.CalMetric(obj.app.dropC(1).Value)],'-k.','LineWidth',1.5,'MarkerSize',10,{'Number of function evaluations',strrep(obj.app.dropC(1).Value,'_',' '),[]});
                            obj.app.waittip.Visible = 'off';
                            isMetric = true;
                    end
                    if ~isMetric
                        obj.app.dropD(2).Value = 'runtime';
                        obj.app.labelD.Text = sprintf('%.4fs',ALG.CalMetric('runtime'));
                    else
                        obj.app.dropD(2).Value = obj.app.dropC(1).Value;
                        value = ALG.CalMetric(obj.app.dropD(2).Value);
                        obj.app.labelD.Text = sprintf('%.4e',value(end));
                    end
                else            % Single-objective optimization
                    switch obj.app.dropC(2).Value
                        case 'Population (variables)'
                            PRO.DrawDec(ALG.result{index,2});
                        otherwise
                            Draw([cell2mat(ALG.result(:,1)),ALG.CalMetric(obj.app.dropC(2).Value)],'-k.','LineWidth',1.5,'MarkerSize',10,{'Number of function evaluations',strrep(obj.app.dropC(2).Value,'_',' '),[]});
                            isMetric = true;
                    end
                    if ~isMetric
                        obj.app.dropD(3).Value = 'runtime';
                        obj.app.labelD.Text = sprintf('%.4fs',ALG.CalMetric('runtime'));
                    else
                        obj.app.dropD(3).Value = obj.app.dropC(2).Value;
                        value = ALG.CalMetric(obj.app.dropD(3).Value);
                        obj.app.labelD.Text = sprintf('%.4e',value(end));
                    end
                end
            end
        end
        %% Create the gif
        function cb_toolbutton1(obj,~,~)
            if ~isempty(obj.app.dropD(1).Items)
                [file,folder] = uiputfile({'*.gif','GIF image'},'');
                figure(obj.platmetaxGUI.app.figure);
                if file ~= 0
                    try
                        filename = fullfile(folder,file);
                        figure('NumberTitle','off','Name','Figure for creating the gif');
                        for i = linspace(0,1,20)
                            obj.app.slider.Value = i;
                            obj.cb_slider([],[],gca);
                            drawnow('limitrate');
                            [I,map] = rgb2ind(print('-RGBImage'),20);
                            if i == 0
                                imwrite(I,map,filename,'gif','Loopcount',inf,'DelayTime',0.2);
                            else
                                imwrite(I,map,filename,'gif','WriteMode','append','DelayTime',0.2);
                            end
                        end
                        delete(gcf);
                    catch
                        uialert(obj.platmetaxGUI.app.figure,sprintf('Fail to save the gif to %s.',filename),'Error');
                        return;
                    end
                end
            end
        end
        %% Open in new figure
        function cb_toolbutton2(obj,~,~)
            if ~isempty(obj.app.dropD(1).Items)
                if isempty(get(gcf,'CurrentAxes'))
                    axes('FontName',obj.app.axes.FontName,'FontSize',obj.app.axes.FontSize,'NextPlot',obj.app.axes.NextPlot,'Box',obj.app.axes.Box,'View',obj.app.axes.View);
                    copyobj(obj.app.axes.Children,gca);
                else
                    h = copyobj(obj.app.axes.Children,gca);
                    for i = 1 : length(h)
                        if strcmp(h(i).Type,'line')
                        	set(h(i),'Color',rand(1,3),'Markerfacecolor',rand(1,3));
                        end
                    end
                end
                axis tight;
                Data = arrayfun(@(s){s.XData,s.YData,s.ZData},get(gca,'Children'),'UniformOutput',false);
                assignin('base','Data',cat(1,Data{:}));
            end
        end
        %% Change the selected result
        function cb_dropdown1(obj,~,~)
            if ~isempty(obj.app.dropD(1).Items)
                obj.cb_slider();
                obj.app.textD.Value = obj.data{obj.app.dropD(1).Value,3};
                if obj.data{obj.app.dropD(1).Value,2}.M > 1
                    set([obj.app.dropC(1),obj.app.dropD(2)],'Visible',true);
                    set([obj.app.dropC(2),obj.app.dropD(3)],'Visible',false);
                else
                    set([obj.app.dropC(1),obj.app.dropD(2)],'Visible',false);
                    set([obj.app.dropC(2),obj.app.dropD(3)],'Visible',true);
                end
            end
        end
        %% Change the selected metric
        function cb_dropdown2(obj,~,~)
            if ~isempty(obj.app.dropD(1).Items)
                if obj.data{obj.app.dropD(1).Value,2}.M > 1
                    if strcmp(obj.app.dropD(2).Value,'runtime')
                        obj.app.dropC(1).Value = obj.app.dropC(1).Items{1};
                    elseif ~strcmp(obj.app.dropD(2).Value,'runtime')
                        obj.app.dropC(1).Value = obj.app.dropD(2).Value;
                    end
                else
                    if strcmp(obj.app.dropD(3).Value,'runtime')
                        obj.app.dropC(2).Value = obj.app.dropC(2).Items{1};
                    elseif ~strcmp(obj.app.dropD(3).Value,'runtime')
                        obj.app.dropC(2).Value = obj.app.dropD(3).Value;
                    end
                end
                obj.cb_slider();
            end
        end
    end
end