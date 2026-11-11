% Example: optimization of the HSSR-SLOPE parameters
%
% The table lists, for each combination of Reconstruction Order (RO),
% Correlation Tolerance (CT) and Harmonics Tolerance (HT), the response
% variables measured against a reference ECG: quartiles of the beat
% detection F1-score and limits of agreement (LOA) of five HRV features.

clear
close all
clc

%% Sensitivity analysis table

tbl = readtable(fullfile("..","data","sensitivity_walking.csv"));

x = tbl{:,["RO","CT","HT"]};
y = tbl{:,4:end};
response_names = string(tbl.Properties.VariableNames(4:end));

%% Optimization setup

% F1-score quartiles are maximized, LOA widths are minimized
options.direction = ["max","max","max","min","min","min","min","min"];

% Importance of each response variable. Change these values to give
% priority to the features that matter for the application, e.g. raise the
% last two (LOA of LF and HF) for a frequency-domain analysis.
options.importance = [1,2,1,3,3,3,3,3];

% Denser grid of parameters for the evaluation of the fitted models
space_RO = 32:4:320;
space_CT = 0:0.005:0.05;
space_HT = 0.1:0.01:0.5;
options.grid = {space_RO,space_CT,space_HT};

%% Optimization

[x_opt,results] = HSSR_Optimizer(x,y,options);

fprintf("\nOptimized parameters for HSSR-SLOPE\n")
fprintf("\nReconstruction Order: %d",x_opt(1))
fprintf("\nCorrelation Tolerance: %0.3f",x_opt(2))
fprintf("\nHarmonics Tolerance: %0.2f\n",x_opt(3))

fprintf("\nPredicted response variables\n")
for i = 1:length(response_names)
    fprintf("\n%s: %0.3f",response_names(i),results.y_opt(i))
end
fprintf("\n")

%% Contour plot of the Global Desirability (fixed Correlation Tolerance)

fixed_CT = 0.025;
[~,idx_CT] = min(abs(space_CT-fixed_CT));

GD = reshape(results.GD,length(space_HT),length(space_CT),length(space_RO));
GD = squeeze(GD(:,idx_CT,:));

figure('Color','w')
contourf(space_RO,space_HT,GD,'LevelStep',0.01,'LineStyle','none')
hold on
contour(space_RO,space_HT,GD,'LevelStep',0.1,'ShowText','on','LineColor','k')
xlabel("Reconstruction Order")
ylabel("Harmonics Tolerance")
xticks(32:32:320)
colorbar
