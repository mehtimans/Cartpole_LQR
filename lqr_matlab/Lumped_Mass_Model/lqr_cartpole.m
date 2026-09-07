clc
clear

%% [1] linearization about the equilibrium point
%%
syms m M L g d
syms x1 x2 x3 x4 
syms u  

Parameter = [m M L g d];
Parameter_value = [0.075 0.694 0.38 9.8065 0.7];
xe = [0 0 pi 0];
x = [x1 x2 x3 x4];

D = m * L^2 * (M + m * (1 - cos(x3)^2));

F = [x2;
    (1/D) * (-m^2*L^2*g*cos(x3)*sin(x3) + m*L^2*(m*L*x4^2*sin(x3)...
      - d*x2) + m*L^2*u);
     x4;
    (1/D) * ((m+M)*m*g*L*sin(x3) - m*L*cos(x3)*(m*L*x4^2*sin(x3)...
      - d*x2) - m*L*cos(x3)*u) 
     ];

G = [x1;
     x3]; 
    
A = sym(zeros(4, 4));
B = sym(zeros(4, 1));
C = sym(zeros(2, 4));
D = zeros(2, 1);

for i = 1: 4
    for j = 1: 4
       A (i, j) = vpa(diff (F(i), x(j)));
    end
end

A_par = (vpa(subs(A, [x, u], [xe, 0]), 10));
A = double(vpa(subs(A, [x, Parameter, u], [xe, Parameter_value, 0]), 10));
disp(A_par)
disp(A)


for i = 1: 4
        B (i, 1) = vpa(diff (F(i), u));
end

B_par = (vpa(subs(B, [x, u], [xe, 0]), 10));
B = double(vpa(subs(B, [x, Parameter, u], [xe, Parameter_value, 0]), 10));
disp(B_par)
disp(B)

for i = 1: 2
    for j = 1: 4
        C (i, j) = vpa(diff (G(i), x(j)));
    end 
end   
C = double(C);

%% LQR controller
%%

Q = diag([1.1, 0.9, 1.5, 0.9]);
R = diag(0.001);

[K_LQR,~,P_LQR] = lqr(A, B, Q, R, 0);
disp(K_LQR)


