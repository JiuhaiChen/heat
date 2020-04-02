clc;clear all
% set the grid size
Jx = 100; dx = 10./Jx;
Jy = 100; dy = 10./Jy;
x = dx*(0:Jx);
y = dy*(0:Jy);


%{
[X1,X2] = meshgrid(x,y);
XX = [X1(:) X2(:)];
mu_1 = [.3 .5];
sigma_1 = [0.25 0.3; 0.3 1]/100;
mu_2 = [.7 .5];
sigma_2 = [0.25 0.3; 0.3 1]/100;
U = mvnpdf(XX,mu_1,sigma_1)/100 + mvnpdf(XX,mu_2,sigma_2)/100;
U = reshape(U,length(x),length(y));
%}

[X1,X2] = meshgrid(x,y);
XX = [X1(:) X2(:)];
mu_1 = [3 5];
sigma_1 = [0.25 0.3; 0.3 1];
mu_2 = [7 5];
sigma_2 = [0.25 0.3; 0.3 1];
U = mvnpdf(XX,mu_1,sigma_1)+mvnpdf(XX,mu_2,sigma_2);
U = reshape(U,length(x),length(y));

figure
surfc(x, y, U)
view(2)
xlabel('X')
ylabel('Y')
zlabel('Concetration')
drawnow

% define a new variable to store new values of U;
Unew=U;
dt = 0.00001;    % dt is extremely small
T = .05;
mu1 = dt/(2*dx); 
a = 1;  % advection 
b = 0.1;    % diffusion
% mu2 = 0.05*dt/(dx*dx);
mu2 = b*dt/(dx*dx);
nt = round(T/dt); 
T = zeros(Jx+1, Jy+1, nt);



for n = 1:nt

    for l = 2:Jy
        for j = 2:Jx
            Unew(j,l) = mu2*(U(j-1,l)-2*U(j,l)+U(j+1,l)) + mu2*(U(j,l-1)-2*U(j,l)+U(j,l+1)) - mu1*a*(U(j+1,l)-U(j-1,l)) - mu1*a*(U(j,l+1)-U(j,l-1)) + U(j,l);
        end
    end

    for l = 2:Jy
        j = 1;
        Unew(j,l) = mu2*(U(Jx,l)-2*U(j,l)+U(j+1,l)) + mu2*(U(j,l-1)-2*U(j,l)+U(j,l+1)) - mu1*a*(U(j+1,l)-U(Jx,l)) - mu1*a*(U(j,l+1)-U(j,l-1)) + U(j,l);
        Unew(Jx+1,l) = U(1,l);
    end

    l = 1;
    for j = 2:Jx      
        Unew(j,l) = mu2*(U(j-1,l)-2*U(j,l)+U(j+1,l)) + mu2*(U(j,Jy)-2*U(j,l)+U(j,l+1)) - mu1*a*(U(j+1,l)-U(j-1,l)) - mu1*a*(U(j,l+1)-U(j,Jy)) + U(j,l);
        Unew(j,Jy+1) = Unew(j,l);
    end

    l = 1;
    j = 1;
    Unew(j,l) = mu2*(U(Jx,l)-2*U(j,l)+U(j+1,l)) + mu2*(U(j,Jy)-2*U(j,l)+U(j,l+1)) - mu1*a*(U(j+1,l)-U(Jx,l)) - mu1*a*(U(j,l+1)-U(j,Jy)) + U(j,l);

    U = Unew;
    T(:,:,n) = Unew;
    
   
end
T = T(:,:,1:100);
figure
surfc(x, y, U)
view(2)
drawnow
% pause(0.05)
save('advection-dispersion','T','x','y','dt','dx','dy')
