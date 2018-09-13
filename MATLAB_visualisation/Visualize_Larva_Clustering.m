%% start of the Gui
function varargout = Visualize_Larva_Clustering(varargin)
% VISUALIZE_LARVA MATLAB code for Visualize_Larva.fig

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Visualize_Larva_Clustering_OpeningFcn, ...
                   'gui_OutputFcn',  @Visualize_Larva_Clustering_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

%% Opens Fcn
function Visualize_Larva_Clustering_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
% handles = define_popo_up_menu(handles);
guidata(hObject,handles );
handles              = capture_figure_properties(handles);

guidata(hObject, handles);


%% Useless Stuff (For us)
function varargout = Visualize_Larva_Clustering_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


%%              Load datas and load trajectories and some parameters


%% Load the .mat file
function Load_Callback(hObject, eventdata, handles)




[FileName,PathName]     = uigetfile('*.mat','Select mat file');
fprintf('%s\n', FileName);
fullpathname            = strcat(PathName, FileName);
handles.fullpathname    = fullpathname;
handles.pathname        = PathName;
handles.name            = FileName;
poule                   = load(fullpathname);
name_loc                = fieldnames(poule);
%%
trx                     = poule.(name_loc{1});
n_trx                   = length(trx);
handles.n_trx           = n_trx;
rig                     = 't5';
% trx                     = generate_new_estimators(trx, rig);

for i =1 : n_trx
    trx(i).global_state_clustering_original = trx(i).global_state_clustering;
    trx(i).global_state_clustering_tagged   = nan(length(trx(i).global_state_clustering),1);
end

% trx = generate_all_estimators_one_shot(trx, rig);


% path_to_name_parameters = '/Users/jbmasson/Dropbox (HHMI)/Path_to_all_programs_sync_hhmi/Matlab/projet/Larva/pipeline_from_choreography_full/Rotta_Choreography/tagged_last_layer_conditional';

% path_to_name_parameters = '/Users/jbmasson/Dropbox (HHMI)/Path_to_all_programs_sync_hhmi/Matlab/projet/Larva/pipeline_from_choreography_full/Rotta_Choreography/tagged_last_layer_conditional_purely_roll';
% poule                   = load([path_to_name_parameters '/name_parameters.mat']);
% name_loc                = fieldnames(poule);
% name_parameters         = poule.(name_loc{1});




handles.larva_number    = n_trx;
handles.trx             = trx;
% handles.name_parameters = name_parameters;
% handles.tagging         = 0;

handles.tagging     = 1;
handles.time_index  = 1;
handles.ind         = 1;
set_text_number(handles);

guidata(hObject,handles );
handles              = get_parameters_out(handles);
guidata(hObject,handles );
handles              = boundaries(handles);
guidata(hObject,handles );
handles              = initialize_some_handles(handles);
guidata(hObject,handles );
handles              = plot_all_figures(handles);
%set_text_roll(handles);
msgbox('file loaded');


guidata(hObject,handles );

%%
function Tag_Load_Classifier_Callback(hObject, eventdata, handles)
% hObject    handle to Tag_Load_Classifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName]  = uigetfile('*.mat','Select Classifier Mat file');
fullpathname         = strcat(PathName, FileName);
handles.fullpathname = fullpathname;
load(fullpathname);
msgbox('file loaded');

handles.classifier   = classifier;

guidata(hObject,handles );

%% chose one of the trajectories
% function Choose_trajectory_Callback(hObject, eventdata, handles)
% 
% % handles.tagging     = 0;
% handles.tagging      = 1;
% handles.time_index   = 1;
% handles              = plot_all_figures(handles);
% set_text_number(handles);
% guidata(hObject,handles );



%%
%% Capture the Index
% function Larvae_Index_Callback(hObject, eventdata, handles)
% 
% items = get(hObject,'String');
% index_selected = get(hObject,'Value');
% item_selected = items{index_selected};
% handles.ind   = str2num(item_selected);
% 
% % modif tagging
% % handles.tagging       = 1;
% handles.visited_trajs = [handles.visited_trajs; handles.ind];
% %disp(handles.visited_trajs);
% set_text_number(handles);
% guidata(hObject,handles );
%%
%% Index Fcn
% function Larvae_Index_CreateFcn(hObject, eventdata, handles)
% 
% if ispc && isequal(get(hObject,'BackgroundColor'), gevariablt(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end
%%
%% define the popup menu
% function handles = define_popo_up_menu(handles)
% 
% n_loc = 500; 
% 
% for i=1:n_loc
%     i_num  = num2str(i);
%     text{i} = i_num;
% end
% set(handles.Larvae_Index,'String',text);
%%
%% initialize some handles
function handles  = initialize_some_handles(handles)


handles.ind                     = 1;
handles.time_index              = 1;
handles.tagging                 = 1;
handles.fast                    = 0;
handles.Play_Forward            = 0;
handles.Play_Backward           = 0;
handles.visited_trajs           = [1];
handles.n_points                = 1e5;
handles.n_depth                 = 1;
handles.boolean_equilibrate     = 1;
%%
%%
%% if checked show trajectory faster
function checkbox_Fast_Callback(hObject, eventdata, handles)

handles.fast = get(hObject,'Value');
guidata(hObject,handles );
%%
%% create the list of features 
function handles = create_list_feature_state(handles)


%trx                       = handles.trx;
% name_feature_classifier   = handles.name_feature_classifier;

    name_parameters = handles.name_parameters  ;
    handles.list_of_variables_to_be_used_to_classify  = name_parameters;
    handles.n_list_of_variables_to_be_used_to_classify = length(handles.list_of_variables_to_be_used_to_classify );   

%     
   
%%
%%                              Start and Stop Tagging

%%
%% Start the Tagging
% function Start_Callback(hObject, eventdata, handles)
% 
% handles.tagging = 1;
% handles = plot_all_figures(handles);
% guidata(hObject,handles );
% %%
% %% Stop the Tagging
% function STOP_Callback(hObject, eventdata, handles)
% 
% handles.tagging = 0;
% handles = plot_all_figures(handles);
% guidata(hObject,handles );
%%
%%
%%                               continuous Tagging
%%
%%
%% play the trajectory backward
function play_forward_Callback(hObject, eventdata, handles)


while (get(hObject,'Value'))
    if (handles.fast)
            pause(0.001);
        else
            pause(0.05); 
    end
    if (handles.time_index >=handles.trx(handles.ind).n)
          handles.time_index=   handles.trx(handles.ind).n;
          handles = plot_all_figures(handles);
    else
        handles.time_index = handles.time_index + 1;
        handles = plot_all_figures(handles);
    end

    guidata(hObject,handles );
end
%%
%% play the trajectory backward
function play_backward_Callback(hObject, eventdata, handles)

while (get(hObject,'Value') )
    if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
    if (handles.time_index <= 1)
        
        handles.time_index  = 1;
        handles = plot_all_figures(handles);

    else
        handles.time_index = handles.time_index - 1;
        handles = plot_all_figures(handles);
    end

    guidata(hObject,handles );
end
%%
%%
% --- Executes on button press in Right. is toggle now
function Right_Callback(hObject, eventdata, handles)


if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).state(handles.time_index) = 1;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end
%%
%%
% --- Executes on button press in Left. is toggle now
function Left_Callback(hObject, eventdata, handles)
% hObject    handle to Left (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
    while (get(hObject,'Value') )
        handles.trx(handles.ind).state(handles.time_index) = -1;
        handles = plot_all_figures(handles);
        if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
        
        guidata(hObject,handles );
    end
end
%%
%%
% --- Executes on button press in Not_Sure. is toggle now
function Not_Sure_Callback(hObject, eventdata, handles)
% hObject    handle to Not_Sure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
    while (get(hObject,'Value') )
        handles.trx(handles.ind).state(handles.time_index) =0;
        handles = plot_all_figures(handles);
        if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
        
        guidata(hObject,handles );
    end
end
%%

%%                                  Step Tagging
%%
%%
%% Go Back one step in time
function Back_Callback(hObject, eventdata, handles)

if ( (handles.time_index == 1) )
    handles.time_index = handles.time_index;
else
    handles.time_index = handles.time_index - 1;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
guidata(hObject,handles );
%%
%% Go Forward one step in time
function Forward_Callback(hObject, eventdata, handles)

if ( (handles.time_index == handles.trx(handles.ind).n) )
    handles.time_index = handles.time_index;
else
    handles.time_index = handles.time_index + 1;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
guidata(hObject,handles );
%%
%% 
function Left_Step_Callback(hObject, eventdata, handles)

% --- Executes on button press in Left_Step.
% hObject    handle to Left_Step (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
    handles.trx(handles.ind).state(handles.time_index) =-1;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
guidata(hObject,handles );
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end

guidata(hObject,handles );
%%
%% not sure state for one step
function Not_Sure_Step_Callback(hObject, eventdata, handles)

if (handles.tagging == 1)
    handles.trx(handles.ind).state(handles.time_index) =0;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index >= handles.trx(handles.ind).n)
            handles.time_index =  handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );
%%
%% 
function Right_Step_Callback(hObject, eventdata, handles)


if (handles.tagging == 1)
   handles.trx(handles.ind).state(handles.time_index) =1;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );
%%
%%
%%
%%                                  Plots
%%
%% boundaries for plots
function handles = boundaries(handles)

variable1 = handles.variable1;
variable2 = handles.variable2;
variable3 = handles.variable3;


for ind = 1: length(handles.trx)
    
% t = handles.t;
handles.trx(ind).t_bounday_min = min(handles.trx(ind).t);
handles.trx(ind).t_bounday_max = max(handles.trx(ind).t);



handles.trx(ind).parameter1_bounday_min = min(handles.trx(ind).(variable1)(:));
handles.trx(ind).parameter1_bounday_max = max(handles.trx(ind).(variable1)(:));

%  
handles.trx(ind).parameter2_bounday_min = min(handles.trx(ind).(variable2)(:));
handles.trx(ind).parameter2_bounday_max = max(handles.trx(ind).(variable2)(:));

%  
handles.trx(ind).parameter3_bounday_min = min(handles.trx(ind).(variable3)(:));
handles.trx(ind).parameter3_bounday_max = max(handles.trx(ind).(variable3)(:));


end
%%
%% capture figure properties 
function handles = capture_figure_properties(handles)

axesHandle1  = findobj(gcf,'Tag','Main_Figure');
set(axesHandle1,'Tag','Main_Figure');
handles.axesHandle1 = axesHandle1;
%plot(handles.axesHandle1, 1, 1, '.k', 'Markersize',40);

axesHandle2  = findobj(gcf,'Tag','Some_Estimator_1');
set(axesHandle2,'Tag','Some_Estimator_1');
handles.axesHandle2 = axesHandle2;

axesHandle3  = findobj(gcf,'Tag','Zoom_Larva');
set(axesHandle3,'Tag','Zoom_Larva');
handles.axesHandle3 = axesHandle3;

% axesHandle6  = findobj(gcf,'Tag','Zoom_Larva_3rd_layer_state');
% set(axesHandle6,'Tag','Zoom_Larva');
% handles.axesHandle6 = axesHandle6;


axesHandle4  = findobj(gcf,'Tag','Some_Estimator_2');
set(axesHandle4,'Tag','Some_Estimator_2');
handles.axesHandle4 = axesHandle4;

axesHandle5  = findobj(gcf,'Tag','Some_Estimator_3');
set(axesHandle5,'Tag','Some_Estimator_3');
handles.axesHandle5 = axesHandle5;



%%
%% plot all figures
function handles = plot_all_figures(handles)


handles = plot_figure_larve_et_trajectoire(handles);
handles = plot_figure_estimator_1(handles);
handles = plot_figure_zoom_larva(handles);
handles = plot_figure_estimator_2(handles);
handles = plot_figure_estimator_3(handles);


% handles = plot_figure_estimator_2(handles);
%%
%%
%% plot the trajectory and the shape of the larvae
function handles = plot_figure_larve_et_trajectoire(handles)


%axes(handles.Main_Figure);
% global_state = handles.global_state;
time_index = handles.time_index;
ind        = handles.ind;
colour     = handles.colour;

hold(handles.axesHandle1, 'off');
plot(handles.axesHandle1,handles.trx(ind).x_center, handles.trx(ind).y_center, '.-', 'Markersize',3, 'color',handles.grey1);

hold(handles.axesHandle1, 'on');

set_text_time(handles, handles.trx(ind).t(time_index))
%disp(handles.trx(ind).state(time_index));
global_state_loc     = handles.trx(ind).global_state(time_index);
global_state_clustering_loc     = handles.trx(ind).global_state_clustering(time_index);
disp(handles.trx(ind).global_state(time_index))
disp(handles.trx(ind).global_state_clustering(time_index))
% 
% 
% if (global_state_loc<0)
%     plot(handles.axesHandle1,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-m', 'Markersize',20,'LineWidth',5);
% elseif isnan(global_state_loc)
%      plot(handles.axesHandle1,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '.-k', 'Markersize',1,'LineWidth',0.1);
% else
     plot(handles.axesHandle1,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-', 'Markersize',20,'LineWidth',3, 'color', colour{global_state_clustering_loc});
% end

% if global_state_3rd_loc==6
%     plot(handles.axesHandle1,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-', 'Markersize',20,'LineWidth',3, 'color', colour{global_state_3rd_loc});
% end

try
    
   plot(handles.axesHandle1,handles.trx(ind).x_head(time_index), handles.trx(ind).y_head(time_index), '.', 'Markersize',30, 'color','r') ;
   plot(handles.axesHandle1,handles.trx(ind).x_tail(time_index), handles.trx(ind).y_tail(time_index), '.', 'Markersize',30, 'color','g') ;
   
   plot(handles.axesHandle1,handles.trx(ind).x_neck_top(time_index) , handles.trx(ind).y_neck_top(time_index), '.', 'Markersize',20, 'color','m') ;
   plot(handles.axesHandle1,handles.trx(ind).x_neck_down(time_index), handles.trx(ind).y_neck_down(time_index), '.', 'Markersize',20, 'color','y') ;
   
end

% plot(handles.axesHandle1,handles.trx(ind).x_head(handles.time_index), handles.trx(ind).y_head(handles.time_index), '.r', 'Markersize',20);
plot(handles.axesHandle1,handles.trx(ind).x_center(handles.time_index), handles.trx(ind).y_center(handles.time_index), '.g', 'Markersize',20);
% plot(handles.axesHandle1,handles.trx(ind).x_tail(handles.time_index), handles.trx(ind).y_tail(handles.time_index), '.b', 'Markersize',20);
if (handles.tagging == 0)
    axis(handles.axesHandle1,[handles.trx(ind).mini_x_tot handles.trx(ind).maxi_x_tot handles.trx(ind).mini_y_tot handles.trx(ind).maxi_y_tot]);
elseif (handles.tagging == 1)
     axis(handles.axesHandle1,[handles.trx(ind).mini_x(time_index) handles.trx(ind).maxi_x(time_index) handles.trx(ind).mini_y(time_index) handles.trx(ind).maxi_y(time_index)]);
     %axis equal;
end
axis square;







%%
%% plot the first estimator
function handles = plot_figure_estimator_1(handles)


time_index     = handles.time_index;
ind            = handles.ind;
variable1      = handles.variable1;
% variable1_bis  = handles.variable1_bis;

hold(handles.axesHandle2, 'off');

plot(handles.axesHandle2, handles.trx(ind).t ,handles.trx(ind).(variable1) ,'.-k', 'Markersize',5,'LineWidth',3);
hold(handles.axesHandle2, 'on');
% plot(handles.axesHandle2, handles.trx(ind).t ,handles.trx(ind).(variable1_bis) ,'-b', 'Markersize',5,'LineWidth',4);

hold(handles.axesHandle2, 'on');
plot(handles.axesHandle2, [handles.trx(ind).t(time_index) handles.trx(ind).t(time_index)],[-10 10],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle2, [handles.trx(ind).t(1) handles.trx(ind).t(end)],[handles.trx(ind).(variable1)(time_index) handles.trx(ind).(variable1)(time_index)],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle2, handles.trx(ind).t(time_index), handles.trx(ind).(variable1)(time_index)  ,'.r', 'Markersize',30);
% 
% global_state_3rd_loc    =  handles.trx(ind).global_state_from_3rd_layer(time_index);
% if global_state_3rd_loc == 6
%     plot(handles.axesHandle2, handles.trx(ind).t ,handles.trx(ind).(variable1), '.', 'Markersize',30,'LineWidth',3, 'color', 'y');
% end




% if (handles.tagging == 0)
%     axis(handles.axesHandle2,[handles.trx(ind).t_bounday_min handles.trx(ind).t_bounday_max handles.trx(ind).parameter1_bounday_min handles.trx(ind).parameter1_bounday_max ]);
% elseif (handles.tagging == 1)
    axis(handles.axesHandle2,[handles.trx(ind).mini_t(time_index) handles.trx(ind).maxi_t(time_index)  handles.trx(ind).mini_param_1(time_index)  handles.trx(ind).maxi_param_1(time_index) ]);
% end
%% plot the second estimator
function handles = plot_figure_estimator_2(handles)


time_index = handles.time_index;
ind        = handles.ind;
variable2  = handles.variable2;
% variable2_bis  = handles.variable2_bis;


hold(handles.axesHandle4, 'off');

plot(handles.axesHandle4, handles.trx(ind).t ,handles.trx(ind).(variable2) ,'.-k', 'Markersize',5,'LineWidth',3);
hold(handles.axesHandle4, 'on');
% plot(handles.axesHandle4, handles.trx(ind).t ,handles.trx(ind).(variable2_bis) ,'-b', 'Markersize',5,'LineWidth',4);

hold(handles.axesHandle4, 'on');
plot(handles.axesHandle4, [handles.trx(ind).t(time_index) handles.trx(ind).t(time_index)],[-10 10],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle4, [handles.trx(ind).t(1) handles.trx(ind).t(end)],[handles.trx(ind).(variable2)(time_index) handles.trx(ind).(variable2)(time_index)],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle4, handles.trx(ind).t(time_index), handles.trx(ind).(variable2)(time_index)  ,'.r', 'Markersize',30);



% global_state_3rd_loc    =  handles.trx(ind).global_state_from_3rd_layer(time_index);
% if global_state_3rd_loc == 6
%     plot(handles.axesHandle4, handles.trx(ind).t ,handles.trx(ind).(variable2), '.', 'Markersize',30,'LineWidth',3, 'color', 'y');
% end

% if (handles.tagging == 0)
%     axis(handles.axesHandle4,[handles.trx(ind).t_bounday_min handles.trx(ind).t_bounday_max handles.trx(ind).parameter2_bounday_min handles.trx(ind).parameter2_bounday_max ]);
% elseif (handles.tagging == 1)
    axis(handles.axesHandle4,[handles.trx(ind).mini_t(time_index) handles.trx(ind).maxi_t(time_index)  handles.trx(ind).mini_param_2(time_index)  handles.trx(ind).maxi_param_2(time_index) ]);
% end


function handles = plot_figure_estimator_3(handles)


time_index = handles.time_index;
ind        = handles.ind;
variable3  = handles.variable3;
% variable3_bis  = handles.variable3_bis;


hold(handles.axesHandle5, 'off');

plot(handles.axesHandle5, handles.trx(ind).t ,handles.trx(ind).(variable3) ,'.-k', 'Markersize',5,'LineWidth',3);
hold(handles.axesHandle5, 'on');
% plot(handles.axesHandle5, handles.trx(ind).t ,handles.trx(ind).(variable3_bis) ,'-b', 'Markersize',5,'LineWidth',4);

hold(handles.axesHandle5, 'on');
plot(handles.axesHandle5, [handles.trx(ind).t(time_index) handles.trx(ind).t(time_index)],[-10 10],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle5, [handles.trx(ind).t(1) handles.trx(ind).t(end)],[handles.trx(ind).(variable3)(time_index) handles.trx(ind).(variable3)(time_index)],'.-k', 'Markersize',5,'LineWidth',1);

plot(handles.axesHandle5, handles.trx(ind).t(time_index), handles.trx(ind).(variable3)(time_index)  ,'.r', 'Markersize',30);
% 
% global_state_3rd_loc    =  handles.trx(ind).global_state_from_3rd_layer(time_index);
% if global_state_3rd_loc == 6
%     plot(handles.axesHandle5, handles.trx(ind).t ,handles.trx(ind).(variable3), '.', 'Markersize',30,'LineWidth',3, 'color', 'y');
% end

% if (handles.tagging == 0)
%     axis(handles.axesHandle5,[handles.trx(ind).t_bounday_min handles.trx(ind).t_bounday_max handles.trx(ind).parameter3_bounday_min handles.trx(ind).parameter3_bounday_max ]);
% elseif (handles.tagging == 1)
    axis(handles.axesHandle5,[handles.trx(ind).mini_t(time_index) handles.trx(ind).maxi_t(time_index)  handles.trx(ind).mini_param_3(time_index)  handles.trx(ind).maxi_param_3(time_index) ]);
% end
%% plot zoone on the larva
function handles = plot_figure_zoom_larva(handles)


%axes(handles.Main_Figure);
% global_state = handles.global_state;
time_index = handles.time_index;
ind        = handles.ind;
colour     = handles.colour;

hold(handles.axesHandle3, 'off');
plot(handles.axesHandle3,handles.trx(ind).x_center, handles.trx(ind).y_center, '.-', 'Markersize',3, 'color',handles.grey1);

hold(handles.axesHandle3, 'on');
plot(handles.axesHandle3,handles.trx(ind).x_spine(time_index,:), handles.trx(ind).y_spine(time_index,:), '-', 'LineWidth',2, 'color',handles.grey1);
%disp(handles.trx(ind).state(time_index));

global_state_loc = handles.trx(ind).global_state(time_index);
plot(handles.axesHandle3,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-', 'Markersize',20,'LineWidth',3, 'color', colour{global_state_loc});
% if (global_state_loc<0)
%     plot(handles.axesHandle3,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-m', 'Markersize',20,'LineWidth',5);
% elseif isnan(global_state_loc)
%      plot(handles.axesHandle3,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '.-k', 'Markersize',1,'LineWidth',0.1);
% else
%      plot(handles.axesHandle3,handles.trx(ind).x_contour(time_index,:), handles.trx(ind).y_contour(time_index,:), '-', 'Markersize',20,'LineWidth',3, 'color', colour{global_state_loc});
% end



try
    
   plot(handles.axesHandle3,handles.trx(ind).x_head(time_index), handles.trx(ind).y_head(time_index), '.', 'Markersize',30, 'color','r') ;
   plot(handles.axesHandle3,handles.trx(ind).x_tail(time_index), handles.trx(ind).y_tail(time_index), '.', 'Markersize',30, 'color','g') ;
   
   plot(handles.axesHandle3,handles.trx(ind).x_neck_top(time_index) , handles.trx(ind).y_neck_top(time_index), '.', 'Markersize',20, 'color','m') ;
   plot(handles.axesHandle3,handles.trx(ind).x_neck_down(time_index), handles.trx(ind).y_neck_down(time_index), '.', 'Markersize',20, 'color','y') ;
   
end

% plot(handles.axesHandle3,handles.trx(ind).x_head(handles.time_index), handles.trx(ind).y_head(handles.time_index), '.r', 'Markersize',20);
plot(handles.axesHandle3,handles.trx(ind).x_center(handles.time_index), handles.trx(ind).y_center(handles.time_index), '.g', 'Markersize',20);
% plot(handles.axesHandle3,handles.trx(ind).x_tail(handles.time_index), handles.trx(ind).y_tail(handles.time_index), '.b', 'Markersize',20);

axis(handles.axesHandle3,[handles.trx(ind).min_x_zoom(time_index) handles.trx(ind).max_x_zoom(time_index) handles.trx(ind).min_y_zoom(time_index) handles.trx(ind).max_y_zoom(time_index)]);
axis off;
% axis equal;


% %% parameters for plot
function handles = get_parameters_out(handles)

trx             = handles.trx;
dx_large        = 12;
dx_small        = 6;

n_past          = 90;
n_future        = 90;
% n_past_param    = 40;
% n_future_param  = 40;
n_past_param    = 20;
n_future_param  = 20;

% define variables
variable1 = 'motion_velocity_norm_smooth_5';
handles.variable1 = variable1;


% variable2 = 'angle_upper_lower_smooth_5_max_min_25';
variable2 = 'S_smooth_5';
handles.variable2 = variable2;

variable3 = 'larva_length_smooth_5';
handles.variable3 = variable3;

handles.grey1         = [0.3 0.3 0.3];
handles.color_state_1 = [1   0   0];
handles.color_state_2 = [0   1   0];
handles.core  = {'run'      , 'cast'  , 'stop' ,  'hunch','back','roll' };
%% definition figure 
orange = [1 0.49 0]
pink = [0.95 0.61 0.73]
grey = [0.70 0.75 0.71]
brown = [0.59 0.29 0]
purple = [0.44 0.16 0.39]
handles.colour = {'k', 'r', 'g', 'b', 'c', 'y', 'm', orange, pink, grey, brown, purple};


for ind = 1 : length(trx)
    
    x                      = trx(ind).x_center;
    n                      = length(x);
    
    
    for i = 1 : n;
    
           
            min_x(i,1) = trx(ind).x_center(i) - dx_large;
            max_x(i,1) = trx(ind).x_center(i) + dx_large;
            min_y(i,1) = trx(ind).y_center(i) - dx_large;
            max_y(i,1) = trx(ind).y_center(i) + dx_large;
            
            min_x_zoom(i,1) = trx(ind).x_center(i) - dx_small;
            max_x_zoom(i,1) = trx(ind).x_center(i) + dx_small;
            min_y_zoom(i,1) = trx(ind).y_center(i) - dx_small;
            max_y_zoom(i,1) = trx(ind).y_center(i) + dx_small;
            
            
    end
    
    min_x = gaussian_smooth_choreography(min_x, 10);
    max_x = gaussian_smooth_choreography(max_x, 10);
    min_y = gaussian_smooth_choreography(min_y, 10);
    max_y = gaussian_smooth_choreography(max_y, 10);
    
    min_x_zoom = gaussian_smooth_choreography(min_x_zoom, 10);
    max_x_zoom = gaussian_smooth_choreography(max_x_zoom, 10);
    min_y_zoom = gaussian_smooth_choreography(min_y_zoom, 10);
    max_y_zoom = gaussian_smooth_choreography(max_y_zoom, 10);

    
    for i = 1 : n;
    
        n_min_loc = max(1,i-n_past);
        n_max_loc = min(n,i+n_future); 
        
        n_min_loc2 = max(1,i-n_past_param);
        n_max_loc2 = min(n,i+n_future_param); 
        
        
        
        mini_x(i) = min( min_x(n_min_loc:n_max_loc));
        maxi_x(i) = max( max_x(n_min_loc:n_max_loc));
        mini_y(i) = min( min_y(n_min_loc:n_max_loc));
        maxi_y(i) = max( max_y(n_min_loc:n_max_loc));
        
        mini_param_1(i) = min(handles.trx(ind).(variable1)(n_min_loc2:n_max_loc2)) ;
        maxi_param_1(i) = max(handles.trx(ind).(variable1)(n_min_loc2:n_max_loc2)) ;
        
        mini_param_2(i) = min(handles.trx(ind).(variable2)(n_min_loc2:n_max_loc2)) ;
        maxi_param_2(i) = max(handles.trx(ind).(variable2)(n_min_loc2:n_max_loc2)) ;

        mini_param_3(i) = min(handles.trx(ind).(variable3)(n_min_loc2:n_max_loc2)) ;
        maxi_param_3(i) = max(handles.trx(ind).(variable3)(n_min_loc2:n_max_loc2)) ;
        
        if isnan( mini_param_1(i))
            mini_param_1(i) = 0;
            maxi_param_1(i) = 1;
        end
        
        if isnan( mini_param_2(i))
            mini_param_2(i) = 0;
            maxi_param_2(i) = 1;
        end
        
        if isnan( mini_param_3(i))
            mini_param_3(i) = 0;
            maxi_param_3(i) = 1;
        end
        
       
        mini_t(i)       = min(handles.trx(ind).t(n_min_loc2:n_max_loc2));
        maxi_t(i)       = max(handles.trx(ind).t(n_min_loc2:n_max_loc2));
        
    
    end
    
    
    mini_x_tot = min(min_x);
    maxi_x_tot = max(max_x);
    mini_y_tot = min(min_y);
    maxi_y_tot = max(max_y);
 
    handles.trx(ind).n        = n;
    handles.trx(ind).n_past   = n_past;
    handles.trx(ind).n_future = n_future;

    handles.trx(ind).min_x    = min_x;
    handles.trx(ind).min_y    = min_y;
    handles.trx(ind).max_x    = max_x;
    handles.trx(ind).max_y    = max_y;
    
    handles.trx(ind).min_x_zoom    = min_x_zoom;
    handles.trx(ind).min_y_zoom    = min_y_zoom;
    handles.trx(ind).max_x_zoom    = max_x_zoom;
    handles.trx(ind).max_y_zoom    = max_y_zoom;

    handles.trx(ind).mini_x    = mini_x;
    handles.trx(ind).mini_y    = mini_y;
    handles.trx(ind).maxi_x    = maxi_x;
    handles.trx(ind).maxi_y    = maxi_y;
    
    handles.trx(ind).mini_x_tot    = mini_x_tot;
    handles.trx(ind).mini_y_tot    = mini_y_tot;
    handles.trx(ind).maxi_x_tot    = maxi_x_tot;
    handles.trx(ind).maxi_y_tot    = maxi_y_tot;
    
    handles.trx(ind).mini_param_1  = mini_param_1;
    handles.trx(ind).maxi_param_1  = maxi_param_1;
    
    handles.trx(ind).mini_param_2  = mini_param_2;
    handles.trx(ind).maxi_param_2  = maxi_param_2;
    
    handles.trx(ind).mini_param_3  = mini_param_3;
    handles.trx(ind).maxi_param_3  = maxi_param_3;    
    
    
    handles.trx(ind).mini_t  = mini_t;
    handles.trx(ind).maxi_t  = maxi_t;
    
    clear min_x min_y max_x max_y mini_x mini_y maxi_x maxi_y mini_x_tot mini_y_tot maxi_x_tot maxi_y_tot ;
    clear min_x_zoom  max_x_zoom  min_y_zoom  max_y_zoom  mini_param_1 maxi_param_1;
    clear  mini_param_2 maxi_param_2 mini_param_3 maxi_param_3;
    
end
%%
%% control the slider
function slider1_Callback(hObject, eventdata, handles)
% --- Executes on slider movement.
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

ind        = handles.ind;
time_index = handles.time_index;
n          = handles.trx(ind).n;


min_slider = get(hObject,'Min');
max_slider = get(hObject,'Max');
delta_x    = max_slider - min_slider;

valeur     = get(hObject,'Value');

time_index = 1 + floor( (valeur-min_slider)*n/delta_x  );
if (time_index>=n)
    time_index = n;
end


handles.time_index = time_index;
handles = plot_all_figures(handles);
guidata(hObject,handles );
%%
%% fcn slider
function slider1_CreateFcn(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.4 .4 .4]);
end
%%
%% go to beginning of trajectory
function Beginning_Callback(hObject, eventdata, handles)
% --- Executes on button press in Beginning.
% hObject    handle to Beginning (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.time_index = 1;
handles = plot_all_figures(handles);

guidata(hObject,handles );
%%
%% go to end of trajectory
function Button_End_Callback(hObject, eventdata, handles)
% --- Executes on button press in Button_End.
% hObject    handle to Button_End (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.time_index = handles.trx(handles.ind).n;
handles = plot_all_figures(handles);

guidata(hObject,handles );
%%



% generate the feature matrix 
function [Features, State, State_original] = generate_feature_matrix_and_state_vector(handles,trx,  n_depth, visited_trajs)
%%
% ind_num                  = num2str(ind);
% visited_trajs_loc = unique(visited_trajs); 

Features                 = [];
State                    = [];
State_original           = [];

fprintf('visited trajectories %i\n', length(visited_trajs));
for i = 1 :  length(visited_trajs)
    fprintf('%i\n', visited_trajs(i));
end


for i = 1 : length(visited_trajs)
    Features                 = generate_features_matrix_tagged(handles,trx, visited_trajs(i), Features);
    [State, State_original]  = generate_state_vector_tagged(trx, visited_trajs(i), State, State_original);
end

[Features, State, State_original]        = clean (Features, State, State_original );

% 
% 
% name                     = handles.name;
% Features_State           = [Features, State];



% Original_Global_State    = handles.trx(ind).original_global_state;


% save([handles.pathname 'Total_Feature_State_' name(1:end-4) '_' ind_num '.mat'], 'Features_State');
% save([handles.pathname 'Total_Feature_State_' name(1:end-4) '.mat'], 'Features_State');

% save([handles.pathname 'Original_Global_State_' name(1:end-4) '_' ind_num '.mat'], 'Original_Global_State');

%%

%% generate the deature space for the classified data
% function [Features] = generate_features_matrix(handles, trx,  n_depth, ind)
% 
% %% generate the feature space for the chosen classifier
% 
% 
% list_of_variables_to_be_used_to_classify   = handles.list_of_variables_to_be_used_to_classify;
% n_list_of_variables_to_be_used_to_classify = handles.n_list_of_variables_to_be_used_to_classify;
% 
% t         = trx(ind).t;
% n         = length(t);
% feature           = zeros(n-n_depth,n_depth*n_list_of_variables_to_be_used_to_classify);
% for l = 1 : n_list_of_variables_to_be_used_to_classify
%           x = getfield(trx(ind), list_of_variables_to_be_used_to_classify{l});
%           
%           for k = 1 : n-n_depth
%                 feature(k, (l-1)*n_depth+1:l*n_depth) = x(1+(k-1):n_depth+(k-1));
%           end
% end
% 
% Features = feature;
% fprintf('%i\t %i\t  %i\n', ind, length(Features(:,1)),length(Features(1,:)) ); 
%%



function Features = generate_features_matrix_tagged(handles,trx, ind, Features)


list_of_variables_to_be_used_to_classify   = handles.list_of_variables_to_be_used_to_classify;
n_list_of_variables_to_be_used_to_classify = handles.n_list_of_variables_to_be_used_to_classify;

% fprintf('%i\n', n_list_of_variables_to_be_used_to_classify);

t         = trx(ind).t;
n         = length(t);
features   = zeros(n,n_list_of_variables_to_be_used_to_classify);

for l = 1 : n_list_of_variables_to_be_used_to_classify
%     fprintf('liste variable %i\n', l);
    x = trx(ind).(list_of_variables_to_be_used_to_classify{l});
%     x = getfield(trx(ind), list_of_variables_to_be_used_to_classify{l});
% fprintf('%s\n', list_of_variables_to_be_used_to_classify{l});
    features(:,l) = x;
end
%%

 Features = [ Features; features];

size(features);



function [State, State_original]    = generate_state_vector_tagged(trx,  ind, State, State_original)
    
   
%     State     = trx(ind).global_state;
%     State     = State(n_depth+1:end);
%     state_loc       = trx(ind).global_state;
    state_loc          = trx(ind).global_state_tagged;
    state_loc_original = trx(ind).global_state_original;
    
    
    II              = state_loc <0.5;
    state_loc(II)   = nan; 
    State           = [ State; state_loc] ;
     State_original = [ State_original;state_loc_original ];
    
    
    
    
%%
%%


%%
%%
function [Features, State, State_original]        = clean (Features, State, State_original )

II       = ~isnan(State);
State    = State(II);
Features = Features(II, :);
State_original = State_original(II);


%%
%%

%%
%% generate the state vector from the tagged data
function [State]    = generate_state_vector(trx,  n_depth, ind)
    
   
    State     = trx(ind).global_state;
    State     = State(n_depth+1:end);
    II        = State <0.5;
    State(II) = -1; 
 
%%    
%%                                  apply   
%5    
%% classify the subset trained
%%
% --- Executes on selection change in Tag_Classifier.
function Tag_Classifier_Callback(hObject, eventdata, handles)

% Hints: contents = cellstr(get(hObject,'String')) returns Tag_Classifier contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Tag_Classifier
%contents                   = cellstr(get(hObject,'String'));
%handles.classifier_choice  = contents;
%handles.current_classifier = contents{get(hObject,'Value')};


contents                        = cellstr(get(hObject,'String'));
handles.classifier_choice       = contents;
handles.name_classifier = contents{get(hObject,'Value')};
guidata(hObject,handles );


%fprintf('%s\n', handles.current_classifier );
%guidata(hObject,handles );

% --- Executes during object creation, after setting all properties.
function Tag_Classifier_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tag_Classifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%


% --- Executes on selection change in Tag_Choose_set_of_Features.
function Tag_Choose_set_of_Features_Callback(hObject, eventdata, handles)

contents                        = cellstr(get(hObject,'String'));
handles.feature_choice          = contents;
handles.name_feature_classifier = contents{get(hObject,'Value')};


if (strcmp(handles.name_feature_classifier(1:end-1),'Custom' ))
    msgbox('have you define in the .m file the list of the features to be used?');
end

%fprintf('%s\n', handles.current_classifier );
guidata(hObject,handles );


% --- Executes during object creation, after setting all properties.
function Tag_Choose_set_of_Features_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tag_Choose_set_of_Features (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



%%                                  The End


% --- Executes on selection change in Tag_parameter_1.
function Tag_parameter_1_Callback(hObject, eventdata, handles)
% hObject    handle to Tag_parameter_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Tag_parameter_1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Tag_parameter_1
contents                   = cellstr(get(hObject,'String'));
handles.classifier_choice  = contents;
handles.current_classifier = contents{get(hObject,'Value')};

%fprintf('%s\n', handles.current_classifier );
guidata(hObject,handles );


% --- Executes during object creation, after setting all properties.
function Tag_parameter_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tag_parameter_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in Tag_parameter_2.
function Tag_parameter_2_Callback(hObject, eventdata, handles)
% hObject    handle to Tag_parameter_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Tag_parameter_2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Tag_parameter_2


% --- Executes during object creation, after setting all properties.
function Tag_parameter_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tag_parameter_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Tag_Save_S_F.
function Tag_Save_S_F_Callback(hObject, eventdata, handles)
% hObject    handle to Tag_Save_S_F (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% effectively_save_evrything(hObject, eventdata, handles);

% trx                     = handles.trx;
% % name_feature_classifier = handles.name_feature_classifier;
% n_depth                 = handles.n_depth;
% n_points                = handles.n_points;
% boolean_equilibrate     = handles.boolean_equilibrate;
% visited_trajs           = handles.visited_trajs;
% visited_trajs           = unique(visited_trajs);
% %current_classifier      = handles.current_classifier;
% handles                 = create_list_feature_state(handles);
% % ind                     = handles.ind;
% 
% [Features, State, State_original] = generate_feature_matrix_and_state_vector(handles,trx, n_depth, visited_trajs);
% 
% name                     = handles.name;
% Features_State           = [Features, State];
% State_change             = [State_original , State];
% 
% fprintf('%s\n', 'save');
% save([handles.pathname 'Total_Feature_State_' name(1:end-4) '.mat'], 'Features_State', 'State_change');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function effectively_save_evrything(hObject, eventdata, handles)

% 
% trx                     = handles.trx;
% % name_feature_classifier = handles.name_feature_classifier;
% n_depth                 = handles.n_depth;
% n_points                = handles.n_points;
% boolean_equilibrate     = handles.boolean_equilibrate;
% visited_trajs           = handles.visited_trajs;
% visited_trajs           = unique(visited_trajs);
% %current_classifier      = handles.current_classifier;
% handles                 = create_list_feature_state(handles);
% % ind                     = handles.ind;
% 
% [Features, State, State_original] = generate_feature_matrix_and_state_vector(handles,trx, n_depth, visited_trajs);
% 
% name                     = handles.name;
% Features_State           = [Features, State];
% State_change             = [State_original , State];
% 
% fprintf('%s\n', 'save');
% save([handles.pathname 'Total_Feature_State_' name(1:end-4) '.mat'], 'Features_State', 'State_change');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press in RUN_continuuous.
function RUN_continuuous_Callback(hObject, eventdata, handles)
% hObject    handle to RUN_continuuous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 1;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in CAST_CONTINUOUS.
function CAST_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to CAST_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 2;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in STOP_CONTINUOUS.
function STOP_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to STOP_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 3;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in HUNCH_CONTINUOUS.
function HUNCH_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to HUNCH_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 4;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in BACK_CONTINUOUS.
function BACK_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to BACK_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 5;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in ROLL_CONTINUOUS.
function ROLL_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to ROLL_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = 6;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

% --- Executes on button press in RUN_STEP.
function RUN_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to RUN_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =1;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in CAST_STEP.
function CAST_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to CAST_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =2;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in STOP_STEP.
function STOP_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to STOP_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =3;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in HUNCH_STEP.
function HUNCH_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to HUNCH_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =4;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in BACK_STEP.
function BACK_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to BACK_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =5;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in ROLL_STEP.
function ROLL_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to ROLL_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =6;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );

% --- Executes on button press in EXCLUDE_CONTINUOUS.
function EXCLUDE_CONTINUOUS_Callback(hObject, eventdata, handles)
% hObject    handle to EXCLUDE_CONTINUOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of EXCLUDE_CONTINUOUS
if (handles.tagging == 1)

while (get(hObject,'Value') )
    
        handles.trx(handles.ind).global_state_tagged(handles.time_index) = nan;
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in MIROR
function MIROR_continuous_Callback(hObject, eventdata, handles)
% hObject    handle to MIROR_continuous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of MIROR_continuous
if (handles.tagging == 1)

while (get(hObject,'Value') )
        
        handles.trx(handles.ind).global_state_tagged(handles.time_index) =  handles.trx(handles.ind).global_state_original(handles.time_index);
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in EXCLUDE_STEP.
function EXCLUDE_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to EXCLUDE_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =nan;
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );


% --- Executes on button press in MIROR_STEP.
function MIROR_STEP_Callback(hObject, eventdata, handles)
% hObject    handle to MIROR_STEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)
   handles.trx(handles.ind).global_state_tagged(handles.time_index) =  handles.trx(handles.ind).global_state_original(handles.time_index);
end
guidata(hObject,handles );
handles = plot_all_figures(handles);
         pause(0.05);
        if (handles.time_index>=handles.trx(handles.ind).n)
            handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end


guidata(hObject,handles );


% --- Executes on button press in NEXT.
function NEXT_Callback(hObject, eventdata, handles)
% hObject    handle to NEXT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.ind < handles.n_trx)
    handles.ind = handles.ind + 1;
    handles.time_index = 1;
    handles = plot_all_figures(handles);
    handles.tagging                 = 1;
    handles.visited_trajs = [handles.visited_trajs; handles.ind];
%     clear_ground_truth_ind(handles, handles.ind);
else
    handles.ind = handles.n_trx;
    handles.time_index = 1;
    handles = plot_all_figures(handles);
    msgbox('you are at the end .... great!');
    handles.tagging                 = 1;
    handles.visited_trajs = [handles.visited_trajs; handles.ind];
end
%set_text_roll(handles);
set_text_number(handles);
% effectively_save_evrything(hObject, eventdata, handles);
% set_text_number(handles);
% set_text_duration(handles);
% set_text_state(handles);
% set_text_parameters(handles);
guidata(hObject,handles );

% --- Executes on button press in PREVIOUS.
function PREVIOUS_Callback(hObject, eventdata, handles)
% hObject    handle to PREVIOUS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.ind > 1)
    
    handles.ind = handles.ind - 1;
    handles.time_index = 1;
    handles = plot_all_figures(handles);
    handles.tagging                 = 1;
    handles.visited_trajs = [handles.visited_trajs; handles.ind];
else

    
    handles.ind = 1;
    handles.time_index = 1;
    handles = plot_all_figures(handles);
        msgbox('it is the first larva');
    handles.tagging                 = 1;
    handles.visited_trajs = [handles.visited_trajs; handles.ind];
end
%set_text_roll(handles);
set_text_number(handles);
% effectively_save_evrything(hObject, eventdata, handles);
% set_text_number(handles);
% set_text_duration(handles);
% set_text_state(handles);
% set_text_parameters(handles);
guidata(hObject,handles );


%%%%%%%%%%%%%%%%%%%%%
function set_text_number(handles)

ind       = handles.ind;
n_trx     = handles.n_trx;

ind_num   = num2str(ind);
n_trx_num = num2str(n_trx);

message_loc = sprintf('%s of %s', ind_num, n_trx_num);
set(handles.num_larva, 'String', message_loc);

%%%%%%%%%%%%%%%%%%%%%
function set_text_roll(handles)
ind       = handles.ind;
n_trx     = handles.n_trx;

ind_num   = num2str(ind);
n_trx_num = num2str(n_trx);

global_state    =  handles.trx(ind).global_state;
II = sum(global_state==6);
if II>0
    message_loc = sprintf('ROLL 5th');
else
    message_loc = sprintf('NO ROLL 5th');
end


% set(handles.Roll_somewhere, 'String', message_loc);

%%%%%%%%%%%%%%%%%%%%%
function set_text_time(handles, t_loc)

try
t_loc_num = num2str(t_loc);

message_loc = sprintf(' %s',t_loc_num);
set(handles.duration, 'String', message_loc);
end
%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in Global_Miror.
function Global_Miror_Callback(hObject, eventdata, handles)
% hObject    handle to Global_Miror (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

    nn = handles.trx(handles.ind).n;
    for i = 1 : nn
        handles.trx(handles.ind).global_state_tagged(i) = handles.trx(handles.ind).global_state_original(i);
    end
end
handles              = plot_all_figures(handles);
guidata(hObject,handles );


% --- Executes on button press in ERASE_MIROR.
function ERASE_MIROR_Callback(hObject, eventdata, handles)
% hObject    handle to ERASE_MIROR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nn                                           = handles.trx(handles.ind).n;
handles.trx(handles.ind).global_state_tagged = nan(nn,1);
handles              = plot_all_figures(handles);
guidata(hObject,handles );


% --- Executes on button press in Mirror_3rd.
function Mirror_3rd_Callback(hObject, eventdata, handles)
% hObject    handle to Mirror_3rd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.tagging == 1)

while (get(hObject,'Value') )
        
        handles.trx(handles.ind).global_state_tagged(handles.time_index) =  handles.trx(handles.ind).global_state_from_3rd_layer(handles.time_index);
        handles = plot_all_figures(handles);
       if(handles.fast)
            pause(0.01);
        else
            pause(0.05); 
        end
        if (handles.time_index>=handles.trx(handles.ind).n)
         handles.time_index = handles.trx(handles.ind).n;
    
        else
            handles.time_index = handles.time_index +1;
            handles = plot_all_figures(handles);
        end
    
        guidata(hObject,handles );
end
    
end
