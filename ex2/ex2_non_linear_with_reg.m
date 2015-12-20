clear all;
close all;
clc;

%```````````````````````````plot data`````````````````````````````

data=load('ex2data2.txt');
X = data(:, [1, 2]);
Y = data(:, 3);
plotData(X, Y);
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
legend('accepted', 'not accepted');

%```````````````````````feature mapping`````````````````````````
