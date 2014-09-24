%% Create data with outlier

N = 50;

x = linspace(0,1,N);
y = x;

y(1)   = 50;
y(end) = -50;

x = x';
y = y';
x = [ones(size(x)), x];

%% Cross validated error of linear regression

lr_errors = zeros(N,1);

for i = 1:N
    train_indices = [(1:(i-1)), (i+1):N]';
    x_train = x(train_indices,2);
    y_train = y(train_indices);
    x_test = x(i,2);
    y_test = y(i,:);
    
    p = polyfit(x_train, y_train, 1);
    
    y_pred = p(2) + p(1)*x_test;
    
    lr_errors(i) = abs(y_test - y_pred);
end

display(mean(lr_errors));

%% Cross validated error of Theil--Sen

ts_errors = zeros(N,1);

for i = 1:N
    train_indices = [(1:(i-1)), (i+1):N]';
    x_train = x(train_indices,2);
    y_train = y(train_indices);
    x_test = x(i,2);
    y_test = y(i,:);
    
    m = Theil_Sen_Regress(x_train,y_train);
    offset = median(y_train - m*x_train);
    
    y_pred = offset + m * x_test;
    
    ts_errors(i) = abs(y_test - y_pred);
end

display(mean(ts_errors));