clear all
close all
clc
make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.13 0.08], [0.07 0.06], [0.05 0.05]);
if ~make_it_tight,  clear subplot;  end

%% load data
% load advection-dispersion;
load heat
[xx,yy]=meshgrid(x,y);
index = 50;
eps = 0.2;

%% prepare derivative 
q = 2:99;
xx = xx(q,q);
yy = yy(q,q);
[xxx,yyy] = meshgrid(0.3:0.3:9.6,0.3:0.3:9.6);


U = T(q,q,index);
dUdt1 = (T(q,q,index+1) - T(q,q,index-1))/(2*dt);
dUdx = (T(q+1,q,index)-T(q-1,q,index))/(2*dx);
dUdy = (T(q,q+1,index)-T(q,q-1,index))/(2*dy);
d2Udx2 = (T(q+1,q,index)-2*T(q,q,index)+T(q-1,q,index))/(dx^2);
d2Udy2 = (T(q,q+1,index)-2*T(q,q,index)+T(q,q-1,index))/(dy^2);
d2Udxdy = (T(q+1,q+1,index) + T(q-1,q-1,index) ...
          - T(q+1,q-1,index) - T(q-1,q+1,index))/(4*dx*dy);

      
%% candidate pool
id = 3:3:96;
xx = xx(id,id); yy = yy(id,id);
xx = xx(:); yy = yy(:); X = [xx yy];

U = U(id,id); dUdt1 = dUdt1(id,id); dUdx = dUdx(id,id); dUdy = dUdy(id,id);
d2Udx2 = d2Udx2(id,id); d2Udy2 = d2Udy2(id,id); d2Udxdy = d2Udxdy(id,id);
U = U(:); dUdt1 = dUdt1(:); dUdx = dUdx(:); dUdy = dUdy(:);
d2Udx2 = d2Udx2(:); d2Udy2 = d2Udy2(:); d2Udxdy = d2Udxdy(:);


%% pool data
[Theta_true] = pool_data(xx,U,dUdx,dUdy,d2Udx2,d2Udy2,d2Udxdy);
N_star = size(U,1);
N0 = 16;    % intial sample size
n_s = 16;   % batch size
ns_max = 20;    % max iteration times


%% prepare true value
[m_x,m_y] = size(Theta_true);   % true value (l2 norm)
Xi_true = zeros(m_y,1);
Xi_true(11,1) = 1;
Xi_true(12,1) = 1;

Xi_log = zeros(m_y,1);    % true Boolean value (false positive + false negative)
Xi_log(11,1) = 1;
Xi_log(12,1) = 1;

%% record the comparison criteria
trial = 50; % repeat times
error_bic = zeros(trial, ns_max);
error_l0_bic = zeros(trial, ns_max);
error_lasso = zeros(trial, ns_max);
error_l0_lasso = zeros(trial, ns_max);

for rep =1:trial
    dUdt = dUdt1 + eps*randn(size(dUdt1));
    s = 0;
    
    %% intial design
    % N_star_square= reshape(1:N_star,[32,32]);
    % chosen_index = N_star_square(2:7:32,2:7:32);
    ii = [26,6; 10,16; 30,22; 4,30; 8,24; 32,8; 20,13; 14,20; 28,28; 18,26; 11,10; 2,11; 22,4; 5,1; 16,31; 23,18];
    chosen_index = ii(:,1) + (ii(:,2)-1)*32;
    chosen_index = chosen_index(:);   
        
    x0 = X(chosen_index,:);
    u0 = U(chosen_index,:);
              
    %% sequential design
    while (1)
        s = s+1;
        
        %% train GP
        [mu,gradient,hessian,cv] = gp_new(x0,u0);         
        
        %% mean
        u1 = mu(X);
        
        %% gradient      
        du1 = gradient(X);
        dudy = du1(:,1);
        dudx = du1(:,2);
       
        %% hessian 
        d2u1 = hessian(X);
        d2udy2 = d2u1(1:N_star,1);
        d2udx2 = d2u1(N_star+1:2*N_star,2);
        d2udxdy = d2u1(1:N_star,2);
        
        %% leave one out cross validation for GP
        sigma = cv/(std(u0)^2);
  
        %% pool Data
        [Theta] = pool_data(xx,u1,dudx,dudy,d2udx2,d2udy2,d2udxdy);
        
        %% sparse regression (BIC)
        Theta_chosen = Theta_true(chosen_index,:);
        eta = dUdt(chosen_index,:);

        mdl = stepwiselm(Theta_chosen,eta,'Criterion','bic');
        chosen_col = mdl.Formula.InModel(1:end-1);
        Theta_lm = Theta_chosen(:,chosen_col);
        mdl = fitlm(Theta_lm,eta,'Intercept',false);
        tol = mdl.RMSE;
        tol = tol/std(eta);
        cof_bic = table2array(mdl.Coefficients(:,1));
        Xi = zeros(m_y,1);
        Xi(chosen_col) = cof_bic;
        chosen_col_bic = double(chosen_col)';
        error_bic(rep,s) = calError(chosen_col_bic,Xi_log);
        error_l0_bic(rep,s) = norm(Xi-Xi_true,2);
        
        
        %% sparse regression (Lasso)
        [B,FitInfo] = lasso(Theta_chosen,eta,'CV',10);
        idxLambdaMinMSE = FitInfo.IndexMinMSE;
        minMSEModelPredictors = B(:,idxLambdaMinMSE)~=0;
        % minMSEModelPredictors = abs(B(:,idxLambdaMinMSE))>1e-3;
        chosen_col_lasso = double(minMSEModelPredictors);
        cof_lasso = B(:,idxLambdaMinMSE);
        Xi = cof_lasso;
        error_lasso(rep,s) = calError(chosen_col_lasso,Xi_log);
        error_l0_lasso(rep,s) = norm(Xi-Xi_true,2);
        
        
        
        %% max sample size
        if s > ns_max
            break
        end
        
        
        
        %% optimal design
        fprintf('(error_regression, error_gp) = (%.3f, %.3f)\n', tol, sigma)
        [chosen_index, index_plot]=optimal_design(Theta_true,Theta,chosen_index,n_s,sigma,tol,X);
        x0 = X(chosen_index,:);
        u0 = U(chosen_index,:);
          
        fprintf('One shot ends!\n\n')
    
    end   

    
end


subplot(2,2,1)
boxplot(error_l0_bic,'PlotStyle','compact')
title('$$ l_2(\beta) $$ (Forward stepwise + BIC)','interpreter','latex','Fontsize',15)
subplot(2,2,2)
boxplot(error_l0_lasso,'PlotStyle','compact')
title('$$ l_2(\beta) $$ (Lasso)','interpreter','latex','Fontsize',15)


subplot(2,2,3)
boxplot(error_bic,'PlotStyle','compact')
title('$$ \gamma $$ (Forward stepwise + BIC)','interpreter','latex','Fontsize',15)
subplot(2,2,4)
boxplot(error_lasso,'PlotStyle','compact')
title('$$ \gamma $$ (Lasso)','interpreter','latex','Fontsize',15)

saveas(gcf,'lasso','epsc')







