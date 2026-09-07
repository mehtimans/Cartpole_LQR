clc
clear

%% [1] linearization about the equilibrium point
%%
syms m_p m_w M l g d
syms x1 x2 x3 x4 
syms u  

Parameter = [m_p m_w M l g];
Parameter_value = [0.2 0.1 0.694 0.5 9.8065];
xe = [0 0 0 0];
x = [x1 x2 x3 x4];

% l is the length of the pole and L is the distance from COM
% m_p is the weight of the pole and m_w is the weirht of the mass

L = (0.5 * m_p + m_w)/(m_p + m_w) * l;
m = m_p + m_w;
I = (1/3 * m_p + m_w) * l^2;
Delta = (I + m*L^2) * (M + m) - (m*L*cos(x3))^2; 

F = [x2;
    (1/Delta) * ((I+m*L^2)*(u+m*L*sin(x3)*x4^2)-m^2*L^2*g*sin(x3)*cos(x3));
     x4;
    (1/Delta) * ((m+M)*m*g*L*sin(x3) - m*L*cos(x3)*(m*L*x4^2*sin(x3)+u)); 
     ];

G = [x1;
     x3]; 
    
A = sym(zeros(4, 4));
B = sym(zeros(4, 1));
C = sym(zeros(2, 4));
D = zeros(2, 1);

% A = [0 1              0            0;
%      0 0   -(L^2 * m^2 * g)/Del      0;
%      0 0              0            1;
%      0 0  (L * m * g * (M + m))/Del   0];
% 
% B = [0; (I + m * L^2)/Del; 0; -L*m/Del];

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

%% design full rank observer
%%

% e = eig(A);
poles_ob = [-50, -50, -40 + 5i, -40 - 5i];
L1 = place(A.',C.',poles_ob).';
% e_obs = eig(A - L1 * C)

%% design intermediate PD control
%%
KP = 0.15;
KD = 0.015;

%% discretization 
%%

% continuous system
con_sys = ss(A, B, C, D); 

% step time
dt = 0.02; 

% discretization using zero order hold
dis_sys = c2d(con_sys, dt, 'zoh');

Ad = dis_sys.A;
Bd = dis_sys.B;
Cd = dis_sys.C;
Dd = dis_sys.D;

%% synbolic discretization 
%% 
syms m L Delta I
syms dt real
A_sym = [0 1              0               0;
         0 0    -(L^2 * m^2 * g)/Delta      0;
         0 0              0               1;
         0 0  (L * m * g * (M + m))/Delta   0];

B_sym = [0; (I + m * L^2)/Delta; 0; -L*m/Delta];

Ad_sym = expm(A_sym * dt);
Bd_sym = inv(A_sym) * (Ad_sym - eye(size(A_sym))) * B_sym;

disp('Symbolic Ad:')
disp(Ad_sym)

disp('Symbolic Bd:')
disp(Bd_sym)


