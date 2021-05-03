rng(1337,'twister');
h=3;
w=5;
A = rand(h,w);
x = linspace(1,10,w);
y = linspace(1,10,h);
[X,Y] = meshgrid(x,y);
xq = linspace(1,10,w*10);
yq = linspace(1,10,h*10);
[XQ,YQ] = meshgrid(xq,yq);
[~,~,vq] = griddata(X,Y,A,XQ,YQ,'v4');
heatmap(vq);

