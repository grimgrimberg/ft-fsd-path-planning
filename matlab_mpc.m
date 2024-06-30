% Specify the file name
filename = 'C:\Users\yuval\ft-fsd-path-planning\combined_measurements_output.csv';

% Read the CSV file
%data = csvread(filename);
data = readtable(filename);

% Extract vectors from the data
%time = data(:, 1); % Assuming the first column is time
%input1 = data(:, 2); % Assuming the second column is the first input
%input2 = data(:, 3); % Assuming the third column is the second input
% Add more input extraction if there are more inputs in the CSV

% Save the data to a MAT file for use in Simulink
save('data.mat', 'time', 'input1', 'input2'); % Adjust the variables accordingly

disp('Data extracted and saved for Simulink.');

% You can directly load this 'data.mat' file in Simulink and use the data as needed.
