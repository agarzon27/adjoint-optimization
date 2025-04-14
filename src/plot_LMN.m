clear variables
clear global

dpath = './data/';
prefix = 'simple_grad_desc_step_2024-nov-02-17-32-50_000001_';
nmod = 50;
n0 = 50;
nfinal = 200;

filename = [prefix,num2str(n0),'.mat'];
load([dpath,filename],'ds');
s_vec=ds*(0:nmod-1)'; 

L0 = [];
LE = [];
Lcost = [];


for i=n0:nmod:nfinal
    filename = [prefix,num2str(i),'.mat'];
    load([dpath,filename],'L0_vec','LE_vec','Lcost_vec');

    if i == n0
        s = s_vec;
    else
        s = [s; s(end) + ds + s_vec];
    end

    L0 = [L0; L0_vec];
    LE = [LE; LE_vec];
    Lcost = [Lcost; Lcost_vec];
end

figure(1); clf
subplot(3,1,1)
plot(s,Lcost,'.-')
ylabel('L')
grid on
title('M, N, and L as functions of s')

subplot(3,1,2)
plot(s,L0,'.-')
ylabel('M')
grid on

subplot(3,1,3)
plot(s,LE,'.-')
ylabel('N (dV^2 ms/cm^2)')
grid on
xlabel('s (dV/cm)^2')

% Plot initial and final electric field signals
filename=[prefix,num2str(nfinal),'.mat'];
load([dpath,filename],'Et_seed_file','Et','dt','final_time')

nt=float_integer(final_time/dt);
tt=dt*(0:nt);
Et1=Et;

load([dpath,Et_seed_file],'Et')
Et0=Et;

figure(2)
plot(tt,Et0,tt,Et1)
title('E_0(t), E_s(t)')
xlabel('t (ms)')
ylabel('E (dV/cm)')
grid on
legend('E_0','E_s')
