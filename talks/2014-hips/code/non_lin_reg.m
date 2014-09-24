%% Create nonlinear data

N = 200;

x = linspace(0,1,N);
%y = (x-0.1).*(x-0.15).*(x-0.9) * 100;
y = zeros(size(x));
for i = 1:N
    if x(i) <= 0.1
        y(i) = x(i) * 10;
    elseif x(i) >= 0.9
        y(i) = -1 + (x(i) - 0.9) * 10;
    else
        y(i) = 1 - 2 * (x(i) - 0.1) / 0.8;
    end
end

y = y * 10;

x = x';
y = y';

%% Plot data

h = figure;
plot(x, y, 'o');
xlabel('x');
ylabel('y');
title('Non-linear data');
save2pdf('non-linear.pdf', h, 600);

%% Cross validated error of linear regression

lr_errors = zeros(N,1);

for i = 1:N
    train_indices = [(1:(i-1)), (i+1):N]';
    x_train = x(train_indices);
    y_train = y(train_indices);
    x_test = x(i);
    y_test = y(i);
    
    p = polyfit(x_train, y_train, 1);
    
    y_pred = p(2) + p(1)*x_test;
    
    lr_errors(i) = abs(y_test - y_pred);
end

display(mean(lr_errors));
display(std(lr_errors) / sqrt(N));

%% Cross validated error of Theil--Sen

ts_errors = zeros(N,1);

for i = 1:N
    train_indices = [(1:(i-1)), (i+1):N]';
    x_train = x(train_indices);
    y_train = y(train_indices);
    x_test = x(i);
    y_test = y(i);
    
    m = Theil_Sen_Regress(x_train,y_train);
    offset = median(y_train - m*x_train);
    
    y_pred = offset + m * x_test;
    
    ts_errors(i) = abs(y_test - y_pred);
end

display(mean(ts_errors));
display(std(ts_errors) / sqrt(N));

%% Cross validate error of kNN

k = 5;

knn_errors = zeros(N,1);

for i = 1:N
    train_indices = [(1:(i-1)), (i+1):N]';
    x_train = x(train_indices);
    y_train = y(train_indices);
    x_test = x(i);
    y_test = y(i);
    
    idx = knnsearch(x_train, x_test, 'K', k);
    
    y_pred = mean(y_train(idx));
    
    knn_errors(i) = abs(y_test - y_pred);
end

display(mean(knn_errors));
display(std(knn_errors) / sqrt(N));