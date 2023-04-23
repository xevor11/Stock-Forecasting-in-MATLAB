%% Stock Price Prediction

%% Network Training Part

%% Parameters determined by Genetic Algorithm
% Optimal Parameters found
epoch=9000;
learningRate=0.01;
goal=6e-05;
neuron=10;

%% Reading Data

fileID = fopen('Dataset\Microsoft_Train.csv');
fgetl(fileID); 
% delimitting the various sub-fields
C=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

%% Formatting Data

% Opening stock value for the day
Open = cell2mat(C(1,2));
Open = Open';

% Highest stock value for the day
High = cell2mat(C(1,3));
High = High';

% Lowest stock value for the day
Low = cell2mat(C(1,4));
Low = Low';

% Closing stock value for the day
Close = cell2mat(C(1,5));
Close = Close';

%% Taking Simple and Moving Average

% Simple Moving Average for 10 and 50 days
simpleMovAvg_10 = tsmovavg(Open,'s',10);
simpleMovAvg_50 = tsmovavg(Open,'s',50);

% Exponential Moving Average for 10 and 50 days
expMovAvg_10 = tsmovavg(Open,'e',10);
expMovAvg_50 = tsmovavg(Open,'e',50);

% Limiting the vector '50:end' because starting values returned after
% moving average is NaN, because of unavailability of data before 1
expMovAvg_10 = expMovAvg_10(1,50:end);
expMovAvg_50 = expMovAvg_50(1,50:end);
simpleMovAvg_10 = simpleMovAvg_10(1,50:end);
simpleMovAvg_50 = simpleMovAvg_50(1,50:end);
Open = Open(1,50:end);
High = High(1,50:end);
Low = Low(1,50:end);
Close = Close(1,50:end);

%% Setting up the Neural Network

% Input vector of the input variables
Input = [Open; High; Low; simpleMovAvg_10; expMovAvg_10; simpleMovAvg_50; expMovAvg_50];

net=feedforwardnet(neuron,'traingdx');
net.layers{1}.transferFcn = 'purelin';
net.divideFcn ='dividetrain';

% Setting the Parameters
net.trainparam.epochs = epoch;
net.trainparam.goal =goal;
net.trainparam.lr = learningRate;

net = train(net, Input, Close);


%% Stock Price Prediction

%% Training Neural Network

%% Optimal Parameters Found by Genetic Algorithm
num_epochs = 9000;
learn_rate = 0.01;
goal_err = 6e-05;
num_neurons = 10;

%% Reading Data

train_fileID = fopen('Dataset\Microsoft_Train.csv');
fgetl(train_fileID);
% delimitting the various sub-fields
train_data = textscan(train_fileID,'%s %f %f %f %f','delimiter',',');
fclose(train_fileID);

%% Formatting Data

% Opening stock value for the day
open_prices = cell2mat(train_data(1,2));
open_prices = open_prices';

% Highest stock value for the day
high_prices = cell2mat(train_data(1,3));
high_prices = high_prices';

% Lowest stock value for the day
low_prices = cell2mat(train_data(1,4));
low_prices = low_prices';

% Closing stock value for the day
close_prices = cell2mat(train_data(1,5));
close_prices = close_prices';

%% Applying Simple and Moving Average

% Simple Moving Average for 10 and 50 days
simple_ma_10 = tsmovavg(open_prices, 's', 10);
simple_ma_50 = tsmovavg(open_prices, 's', 50);

% Exponential Moving Average for 10 and 50 days
exp_ma_10 = tsmovavg(open_prices, 'e', 10);
exp_ma_50 = tsmovavg(open_prices, 'e', 50);

% Limiting the vector '50:end' because starting values returned after
% moving average is NaN, because of unavailability of data before 1
exp_ma_10 = exp_ma_10(1, 50:end);
exp_ma_50 = exp_ma_50(1, 50:end);
simple_ma_10 = simple_ma_10(1, 50:end);
simple_ma_50 = simple_ma_50(1, 50:end);
open_prices = open_prices(1, 50:end);
high_prices = high_prices(1, 50:end);
low_prices = low_prices(1, 50:end);
close_prices = close_prices(1, 50:end);

%% Setting up the Neural Network

% Input vector of the input variables
Input = [open_prices; high_prices; low_prices; simple_ma_10; exp_ma_10; simple_ma_50; exp_ma_50];

net = feedforwardnet(num_neurons, 'traingdx');
net.layers{1}.transferFcn = 'purelin';
net.divideFcn ='dividetrain';

% Setting the Parameters
net.trainparam.epochs = num_epochs;
net.trainparam.goal = goal_err;
net.trainparam.lr = learn_rate;

net = train(net, Input, close_prices);

%% Validation Test of the constructed neural network
%%

% Opening sample test data
val_fileID = fopen('Dataset\Microsoft_Validation.csv');
fgetl(val_fileID);
val_data = textscan(val_fileID,'%s %f %f %f %f','delimiter',',');
fclose(val_fileID);

% Formatting the Validation Data
openVal = cell2mat(val_data(1, 2));
openVal = openVal';
highVal = cell2mat(val_data(1, 3));
highVal = highVal';
lowVal = cell2mat(val_data(1, 4));
lowVal = lowVal';
closeVal = cell2mat(val_data(1, 5));
closeVal = closeVal';

% Taking Simple and Moving Average
simple_ma_10_val = tsmovavg(openVal, 's', 10);
simple_ma_50_Val = tsmovavg(openVal, 's', 50);
exp_ma_10_Val = tsmovavg(openVal, 'e', 10);
exp_ma_50_Val = tsmovavg(openVal, 'e', 50);

% Limiting the vector '50:end' because starting values returned after
% moving average is NaN, because of unavailability of data before 1
exp_ma_10_Val = exp_ma_10_Val(1, 50:end);
exp_ma_50_Val = exp_ma_50_Val(1, 50:end);
simple_ma_10_val = simple_ma_10_val(1, 50:end);
simple_ma_50_Val = simple_ma_50_Val(1, 50:end);
openVal = openVal(1, 50:end);
highVal = highVal(1, 50:end);
lowVal = lowVal(1, 50:end);
closeVal = closeVal(1, 50:end);

% Validating Network
Input_t = [openVal; highVal; lowVal; simple_ma_10_val; exp_ma_10_Val; simple_ma_50_Val; exp_ma_50_Val];
answer = net(Input_t);

% Calculating Performance Error
retMSE = mse(answer - closeVal);

% Plotting Output
x = 1:numel(closeVal);
plot(x, answer, x, closeVal);
legend('Predicted Value', 'Actual Value', 'Location', 'southeast')
xlabel('Data Points');
ylabel('Closing Stock Market Values');
title('Stock Market Prediction');
grid on

save net.mat