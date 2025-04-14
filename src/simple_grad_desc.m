clear variables
clear global

[status,output]=system('hostname');
if status==0
    hostname=strtrim(output);
    fprintf('hostname="%s"\n',hostname)
else
    error('Hostname not found')
end

disp(['****************start time= ',str_clock])

program=mfilename;

var_list = {'alpha', % weight of electric field integral
            'ds', % pseudo-time step duration
            'ns', % total number of pseudo-time steps
            'final_time',
            'gamma', % weights of gradients of variables
            'nmod', %
            'Et_seed_file',
            'icondfile',
            'grid_file',
            'S_file',
            'mex_function',
            'in_path',
            'out_path',
            'jobid'};

infile = 'infile_grad';
fid = open_file_read(infile);

values=read_check_2(fid,var_list);
fclose(fid);
fprintf('Parameters read from ''%s'' \n',infile)

for i=1:length(var_list)
  eval(sprintf('%s=values{%d};',var_list{i},i));
end
          
dt=0.1;

fprintf('alpha=%f\n',alpha)
fprintf('ds=%f\n',ds)
fprintf('ns=%f\n',ns)
fprintf('final_time=%f\n',final_time)
fprintf('gamma=[%.2f, %.2f, %.2f]\n',gamma(1), gamma(2), gamma(3))
fprintf('Et_seed_file=%s\n',Et_seed_file)

nt=float_integer(final_time/dt);
tt=(0:nt)*dt;

load([in_path,Et_seed_file],'Et')

% convert Et to column vector, if necessary.
Et = Et(:);

load([in_path, icondfile],'yf')

load([in_path,grid_file],'active_index','ghost_lr')
bx = (active_index>0);
bx3 = [bx;bx;bx];
y0 = yf(bx3);

nrows=256;
ncols=256;

load([in_path,S_file],'S');

time_str=str_clock_no_mu;
time_str=[time_str,'_',jobid];

Lcost_vec = zeros(nmod,1);
L0_vec = zeros(nmod,1);
LE_vec = zeros(nmod,1);

for i=1:ns
    disp('---------------- COMPUTING GRADIENT -------------')
    tic

    [gradE,Lcost,L0,LE, param_str] = mex_function(Et, dt, nt, y0, active_index, ...
                                                  nrows, ncols, ghost_lr, gamma, S, alpha);

    time = toc;
    fprintf("time=%f\n",time)
    
    ip = mod(i-1,nmod)+1;
    Lcost_vec(ip) = Lcost;
    L0_vec(ip) = L0;
    LE_vec(ip) = LE;

    if mod(i,nmod)==0
        % below, Et is stored *before* it is updated
        outfile=['simple_grad_desc_step_',time_str,'_',num2str(i),'.mat'];
        disp(outfile)
        save([out_path,outfile],'alpha','dt','final_time','Et_seed_file',...
             'Et','gradE','Lcost','ds','Lcost_vec','L0_vec','LE_vec','gamma','program',...
             'hostname','nmod','icondfile','grid_file','S_file','mex_function')
        disp(['file "',out_path,outfile,'" written'])
    end

    Et = Et - ds*gradE;

    disp(['********************* i_grad = ',num2str(i),' of ',num2str(ns)])

end

% Forcing quit through the operating system
% id = feature('getpid');
% cmd = sprintf('kill -9 %d', id);
% system(cmd)
