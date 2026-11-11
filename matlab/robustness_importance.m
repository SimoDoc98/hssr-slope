% Robustness of the selected HSSR-SLOPE parameters to the importance values
%
% For each condition of the sensitivity analysis, the Global Desirability
% (GD) is recomputed for all the 3^8 = 6561 combinations of importance
% values between 1 and 3 assigned to the eight response variables. The GD
% of the selected parameters is compared with the highest GD among the
% tested combinations of Reconstruction Order (RO) and Harmonics Tolerance
% (HT), with Correlation Tolerance fixed at 0.025.
%
% Output: robustness_importance.csv (Supplementary Table 4 of the paper)

clear
close all
clc

%% Setup

data_dir = fullfile("..","data");

files = ["Sensitivity_Analysis_PPGDalia.xlsx","Sensitivity_Analysis_WESAD.xlsx"];
conditions = {["Rest","Cycling","Walking","Working"], ...
              ["Baseline","Meditation","Amusement","Stress"]};

% Parameters selected in the paper (RO, HT), in the same order as above
selected = [128 0.40; 256 0.25; 288 0.15; 224 0.10; ...
            256 0.35; 224 0.35; 256 0.20; 256 0.10];

% Optimization setup (same as in the paper)
options.direction = ["max","max","max","min","min","min","min","min"];
options.importance = [1,2,1,3,3,3,3,3];
options.grid = {32:4:320, 0:0.005:0.05, 0.1:0.01:0.5};

% Tested combinations of the full-factorial design
fixed_CT = 0.025;
tested_RO = 32:32:320;
tested_HT = 0.10:0.05:0.50;

% All combinations of importance values between 1 and 3
w = cell(1,8);
[w{:}] = ndgrid(1:3);
W = cell2mat(cellfun(@(v) v(:),w,'UniformOutput',false)); % 6561 x 8

%% Analysis

results = table();
k = 0;

for f = 1:length(files)
    for c = conditions{f}

        k = k + 1;

        % Sensitivity analysis results
        x = readmatrix(fullfile(data_dir,files(f)),"Range",'B5:D274',"Sheet",c);
        y = readmatrix(fullfile(data_dir,files(f)),"Range",'E5:L274',"Sheet",c);

        % Regression models and predicted responses
        [~,res] = HSSR_Optimizer(x,y,options);

        % Desirability of each response (independent of the importance values)
        d = zeros(size(res.y_hat));
        for i = 1:size(d,2)
            L = min(res.y_hat(:,i)) - 0.01;
            U = max(res.y_hat(:,i)) + 0.01;
            if options.direction(i) == "max"
                d(:,i) = (res.y_hat(:,i)-L)./(U-L);
            else
                d(:,i) = (U-res.y_hat(:,i))./(U-L);
            end
        end

        % Tested combinations of RO and HT at fixed CT
        is_tested = abs(res.x_grid(:,2)-fixed_CT) < 1e-9 & ...
                    any(abs(res.x_grid(:,1)-tested_RO) < 1e-9,2) & ...
                    any(abs(res.x_grid(:,3)-tested_HT) < 1e-9,2);

        x_tested = res.x_grid(is_tested,:);
        log_d = log(d(is_tested,:));

        % Selected combination
        i_sel = find(x_tested(:,1) == selected(k,1) & abs(x_tested(:,3)-selected(k,2)) < 1e-9);

        % GD with the importance values of the paper
        GD_paper = exp(log_d*options.importance'/sum(options.importance));
        [GD_best,i_best] = max(GD_paper);

        % GD with all the combinations of importance values
        GD_all = exp((log_d*W')./sum(W,2)');
        ratio = GD_all(i_sel,:)./max(GD_all,[],1);

        row = table(c,selected(k,1),selected(k,2),x_tested(i_best,1),x_tested(i_best,3), ...
            GD_paper(i_sel),GD_best,100*mean(ratio >= 0.95),median(ratio), ...
            'VariableNames',["Condition","RO_selected","HT_selected","RO_best","HT_best", ...
            "GD_selected","GD_best","Within5pct","MedianRatio"]);
        results = [results;row]; %#ok<AGROW>

        fprintf("%s done\n",c)
    end
end

%% Results

disp(results)
writetable(results,"robustness_importance.csv")
