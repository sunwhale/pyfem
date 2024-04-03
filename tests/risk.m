
pt=12;%字体大小
tol=1e-10;%误差限
num=35;%点的个数
time=3;%绘图时间
dy=0.8/num;%载荷增量
x_newton=zeros(num,1);y_newton=x_newton;
dl=0.1;%弧长
x_arc=zeros(num,1);y_arc=x_arc;


x=0:0.01:3;
[y,~]=arrayfun(@fun,x);
h=figure;figset([20,7,0.3,0.3]);
subplot(1,2,1);plot(x,y,'k','linewidth',1.5);hold on;grid on;
set(gca,'ticklabelinterpreter','latex','fontsize',pt);
xlabel('$a$','interpreter','latex','fontsize',pt);
ylabel('$\lambda$','interpreter','latex','fontsize',pt,'Rotation',0);
title('Newton''s Method','interpreter','latex','fontsize',pt);

subplot(1,2,2);plot(x,y,'k','linewidth',1.5);hold on;grid on;
set(gca,'ticklabelinterpreter','latex','fontsize',pt);
xlabel('$a$','interpreter','latex','fontsize',pt);
ylabel('$\lambda$','interpreter','latex','fontsize',pt,'Rotation',0);
title('Arc Length Method','interpreter','latex','fontsize',pt);

for i=2:num
    [x_newton(i),y_newton(i)]=newtons_method(x_newton(i-1),dy,tol);
    subplot(1,2,1);
    plot(x_newton(i),y_newton(i),'ob','linewidth',1.5,'markersize',7);

    [x_arc(i),y_arc(i)]=arc_method(x_arc(i-1),dl,tol);
    subplot(1,2,2);
    plot(x_arc(i),y_arc(i),'ob','linewidth',1.5,'markersize',7);
    pause(time/num);
end

% out = [y,dy/dx]
function [y,dy]=fun(x)

    st=sin(pi/3);
    temp=1-2*st*x+x^2;
    y=(1/sqrt(temp)-1)*(st-x);
    dy=(st-x)^2/temp^(3/2)-(1/sqrt(temp)-1);

end

% 牛顿法
% input: 初始点x0, 载荷增量dy, 误差限tol
%output: 结束点[x1,y1]
function [x1,y1]=newtons_method(x0,dy,tol)

    xi=x0;
    [yi,dyi]=fun(xi);%初始点
    %dy=0.01;%载荷增量
    y1=yi+dy;%结束点

    Nmax=1000;num=1;
    while 1
        xj=xi+(y1-yi)/dyi;
        [yj,dyj]=fun(xj);
        xi=xj;yi=yj;dyi=dyj;
        num=num+1;
        if (abs(yi-y1)<tol) || (num==Nmax)
            break;
        end
    end
    x1=xi;y1=yi;

end

% 弧长法
% input: 初始点x0, 弧长dl, 误差限tol
%output: 结束点[x1,y1]
function [x1,y1]=arc_method(x0,dl,tol)

    %tol=1e-10;%误差限
    [y0,dy0]=fun(x0);%初始点
    xi=x0;yi=y0;dyi=dy0;
    %dl=0.01;%弧长

    Nmax=1000;num=1;
    while 1
        temp_t=yi-dyi*xi-y0;
        temp_a=1+dyi^2;
        temp_b=2*dyi*temp_t-2*x0;
        temp_c=temp_t^2+x0^2-dl^2;
        delta=temp_b^2-4*temp_a*temp_c;
        if delta<0
            disp('delta<0');
            break;
        end
        xj1=(-temp_b+sqrt(delta))/2/temp_a;
        xj2=(-temp_b-sqrt(delta))/2/temp_a;
        xj=max(xj1,xj2);%取增大项
        [yj,dyj]=fun(xj);
        num=num+1;
        if (abs(yj-yi)<tol) || (num==Nmax)
            %disp(['err=',num2str(abs(yj-yi))]);
            %disp(['radius=',num2str(sqrt((xj-x0)^2+(yj-y0)^2))]);
            break;
        end
        xi=xj;yi=yj;dyi=dyj;
    end
    x1=xi;y1=yi;
    
end

%设置图片尺寸与位置
function figset(parameter1,parameter2)

    %电脑屏幕尺寸
    set(0,'units','centimeters')
    cm_ss=get(0,'screensize');
    W=cm_ss(3);%电脑屏幕长度，单位cm
    H=cm_ss(4);%电脑屏幕宽度，单位cm

    %设置figure在screen中的比例位置与大小
    temp1=[parameter1(3),parameter1(4),parameter1(1)/W,parameter1(2)/H];
    set(gcf,'units','normalized','position',temp1);
    if nargin==2
        %设置axis在figure中的比例位置与大小
        temp2=[parameter2(3),parameter2(4),parameter2(1),parameter2(2)];
        set(gca,'position',temp2);
    end

end