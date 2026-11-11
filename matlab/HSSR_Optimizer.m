function [x_opt,results] = HSSR_Optimizer(x,y,options)
% HSSR_Optimizer  Multi-response optimization of the HSSR-SLOPE parameters.
%
%   [x_opt,results] = HSSR_Optimizer(x,y,options)
%
%   A second-order regression model is fitted to each response variable
%   measured on a full-factorial design of the three HSSR-SLOPE parameters.
%   The models are evaluated on a denser grid, each predicted response is
%   converted into a desirability between 0 and 1, and the Global
%   Desirability (weighted geometric mean of the desirabilities) is
%   maximized (Derringer and Suich, 1980; Vera Candioti et al., 2014).
%
%   Inputs
%     x        n-by-3 matrix of the tested parameters: Reconstruction
%              Order, Correlation Tolerance, Harmonics Tolerance
%     y        n-by-m matrix of the response variables
%     options  struct with fields
%       direction   1-by-m string array, "max" or "min" for each response
%       importance  1-by-m vector, importance of each response
%       grid        1-by-3 cell array with the values of each parameter
%                   on which the models are evaluated
%       alpha       significance level of the regression coefficients
%                   (default: 0.05)
%       verbose     if true, the fitted models are displayed
%                   (default: false)
%
%   Outputs
%     x_opt    parameters with the highest Global Desirability
%     results  struct with fields
%       y_opt   responses predicted at x_opt (scale of the measurements)
%       GD      Global Desirability on the grid
%       x_grid  grid of parameters (one row for each point)
%       y_hat   linear predictor of each response on the grid
%       models  fitted regression models
%
%   Copyright (c) 2025 Simone Costantini, Politecnico di Milano.
%   Distributed under the BSD 2-Clause License (see LICENSE).

if ~isfield(options,"alpha")
    options.alpha = 0.05;
end
if ~isfield(options,"verbose")
    options.verbose = false;
end

N_y = size(y,2);
r = options.importance;

%% Response Surfaces estimation (II order regression model)

distributions = ["normal","gamma"];
link_functions = ["identity","log"];

x1 = x(:,1); % --> Reconstruction Order
x2 = x(:,2); % --> Correlation Tolerance
x3 = x(:,3); % --> Harmonics Tolerance

model = 'yi ~ x1 + x2 + x3 + x1^2 + x2^2 + x3^2 + x1*x2 + x1*x3 + x2*x3';

models = cell(1,N_y);
is_log_link = false(1,N_y);

warning('off','all');

for i = 1:N_y

    yi = y(:,i);
    tbl = table(x1,x2,x3,yi,'VariableNames',{'x1','x2','x3','yi'});

    % Fitting with each candidate distribution
    glm_yij = cell(1,length(distributions));
    AIC = zeros(1,length(distributions));

    for j = 1:length(distributions)
        glm_yij{j} = fitglm(tbl,model,"Distribution",distributions(j),"DispersionFlag",true,"Link",link_functions(j),"Intercept",true);
        AIC(j) = 2*glm_yij{j}.NumEstimatedCoefficients - log(glm_yij{j}.LogLikelihood);
    end

    % Selecting the best distribution
    [~,best_distribution] = min(AIC);

    models{i} = glm_yij{best_distribution};
    is_log_link(i) = link_functions(best_distribution) == "log";

    if options.verbose
        fprintf('\n### Regression model for the response variable %d\n',i);
        disp(models{i});
        fprintf("Adjusted R^2: %0.4f\n",models{i}.Rsquared.Adjusted)
    end
end

warning('on','all');

%% Computing the estimated response variables y_hat

% Denser domain of x
[X3,X2,X1] = ndgrid(options.grid{3},options.grid{2},options.grid{1});
x_grid = [X1(:),X2(:),X3(:)];

y_hat = zeros(size(x_grid,1),N_y);

for i = 1:N_y

    coefficients = models{i}.Coefficients.Estimate;
    p_values = models{i}.Coefficients.pValue;
    coefficient_names = models{i}.CoefficientNames;

    % Only significant coefficients are used for prediction
    for k = 1:length(coefficients)
        if p_values(k) < options.alpha
            y_hat(:,i) = y_hat(:,i) + coefficients(k) * model_term(coefficient_names{k},x_grid);
        end
    end

end

%% Multiple Response Optimization

% Desirability of each response variable
d = zeros(size(y_hat));

for i = 1:N_y
    L = min(y_hat(:,i)) - 0.01;
    U = max(y_hat(:,i)) + 0.01;
    if options.direction(i) == "max"
        d(:,i) = (y_hat(:,i)-L)./(U-L); % the more the better
    elseif options.direction(i) == "min"
        d(:,i) = (U-y_hat(:,i))./(U-L); % the less the better
    else
        error("Direction must be ""max"" or ""min"".")
    end
end

% Global Desirability (GD)
GD = ones(size(y_hat,1),1);
for i = 1:N_y
    GD = GD.*(d(:,i).^r(i));
end
GD = GD.^(1/sum(r));

% Maximum of GD
[~,GD_idx_max] = max(GD);

x_opt = x_grid(GD_idx_max,:);

y_opt = y_hat(GD_idx_max,:);
y_opt(is_log_link) = exp(y_opt(is_log_link));

results.y_opt = y_opt;
results.GD = GD;
results.x_grid = x_grid;
results.y_hat = y_hat;
results.models = models;

end

function v = model_term(name,x_grid)
% Value of a regression term ('(Intercept)', 'x1', 'x1:x3', 'x2^2') on the grid

v = ones(size(x_grid,1),1);

if strcmp(name,'(Intercept)')
    return
end

factors = strsplit(name,':');

for k = 1:length(factors)
    parts = strsplit(factors{k},'^');
    column = str2double(parts{1}(2:end));
    if length(parts) == 2
        v = v .* x_grid(:,column).^str2double(parts{2});
    else
        v = v .* x_grid(:,column);
    end
end

end
