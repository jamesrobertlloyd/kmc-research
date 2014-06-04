%% SE - solar

folder = 'SE';

files = dir(strcat(folder, '/*.mat'));
file = files(2);

fig_title = 'Solar';

plot_data_and_witness;

%% SE - gas production

folder = 'SE';

files = dir(strcat(folder, '/*.mat'));
file = files(9);

fig_title = 'Gas production';

plot_data_and_witness;

%% ABCD - unemployment

folder = 'ABCD';

files = dir(strcat(folder, '/*.mat'));
file = files(11);

fig_title = 'Unemployment';

plot_data_and_witness;

%% ABCD - internet

folder = 'ABCD';

files = dir(strcat(folder, '/*.mat'));
file = files(6);

fig_title = 'Internet';

plot_data_and_witness;