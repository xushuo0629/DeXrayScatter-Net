function [xf, A_max,A_min, A_mean] = subfun_A(x, LL, batch)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
% LL = 4000;
y = x(1:LL);
% batch =200;

sz = [batch, floor(length(y)/batch)];
L = sz(1)*sz(2);
A = reshape(y(1:L),sz);

A_max = max(A);
A_min  = min(A);
A_mean = mean(A);
xf = (0:sz(2)-1)*batch;
end

