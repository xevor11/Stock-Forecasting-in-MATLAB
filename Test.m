%% Network Testing Part

% Load trained network
load net

% Read test data
filename = 'Dataset\Microsoft_Test.csv';
data = readtable(filename);

% Extract input features
Open = data.Open;
High = data.High;
Low = data.Low;
Close = data.Close;
simpleMovAvg_10 = movmean(Open, 10);
expMovAvg_10 = tsmovavg(Open, 'e', 10);
simpleMovAvg_50 = movmean(Open, 50);
expMovAvg_50 = tsmovavg(Open, 'e', 50);

% Limit the vectors to exclude NaN values due to moving averages
start_idx = max([10, 50]);
Open = Open(start_idx:end);
High = High(start_idx:end);
Low = Low(start_idx:end);
Close = Close(start_idx:end);
simpleMovAvg_10 = simpleMovAvg_10(start_idx:end);
expMovAvg_10 = expMovAvg_10(start_idx:end);
simpleMovAvg_50 = simpleMovAvg_50(start_idx:end);
expMovAvg_50 = expMovAvg_50(start_idx:end);

% Prepare input matrix for network
Input = [Open; High; Low; simpleMovAvg_10; expMovAvg_10; simpleMovAvg_50; expMovAvg_50];

% Test network and calculate MSE
predictions = net(Input);
mse = mean((predictions - Close).^2);

% Plot predicted and actual values
plot(predictions);
hold on;
plot(Close);
legend('Predicted Value','Actual Value','Location','southeast');
xlabel('Data Points');
ylabel('Closing Stock Market Value');
title('Stock Market Prediction');
grid on;
hold off;
