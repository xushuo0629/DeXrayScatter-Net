% clc;clear;
%%
tex = importdata('train_loss.txt');
str = tex{1};
x = regexp (str, '[', 'split');
x = regexp (x{2}, ']', 'split');
x = regexp (x{1}, ',', 'split');

x1 = str2num(char(x)); 

%%
LL = length(x1);
batch =8;
[xf , A_max, A_min,  A_mean] = subfun_A(x1, LL, batch);
% [xf, B_max, B_min,  B_mean] = subfun_A(x2, LL, batch);

batch2 = 2;
[xft   , At_max, At_min,  At_mean] = subfun_A(x1, LL, batch2);
% [xft , Bt_max,  Bt_min,  Bt_mean] = subfun_A(x2, LL,  batch2);
%% plot
figure('color','w')
% plot(xf,A_mean,'b-','LineWidth',1.5);
% hold on
% plot(xf,B_mean,'r-','LineWidth',1.5);
% grid on
plot(xft,At_mean,'b-','LineWidth',1.5);
hold on
% plot(xft,Bt_mean,'r-','LineWidth',1.5);
grid on
area1 = fill([xf  fliplr(xf)],[A_min fliplr(A_max)],[0.93333, 0.83529, 0.82353],'edgealpha', '0', 'facealpha', '.5');
% area2 = fill([xf  fliplr(xf)],[B_min fliplr(B_max)],[204, 255, 204]./255,'edgealpha', '0', 'facealpha', '.5');
xlabel('epoch','FontName','Times New Roman','FontSize',12);
ylabel('train loss','FontName','Times New Roman','FontSize',12)
ylim([0, 100])
xlim([0,LL-batch])
%legend( 'train', 'validation','uncertainty of train','uncertainty of validation','FontName','Times New Roman','FontSize',12)
legend( 'train','uncertainty of train','FontName','Times New Roman','FontSize',12)