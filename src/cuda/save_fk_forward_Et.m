clear variables
clear global
program=mfilename;

[status,output]=system('hostname');
if status==0
    hostname=strtrim(output);
    fprintf('hostname="%s"\n',hostname)
else
    error('Hostname not found')
end


disp('###############################')
disp('#                             #')
disp('#   HAVE YOU COMPILED THE     #')
disp('#   MEX FUNCTION?             #')
disp('#                             #')
disp('###############################')
pause(0.5)

var_list = {'Et_file', % file with electric field signal
            'final_time',
            'nstep',% solution will be saved every nstep time steps
            'icondfile',
            'outfile',
            'outfile_prefix'
            'mex_function',
            'grid_file',
            'in_path',
            'out_path'};

infile = 'infile_forward';
fid = open_file_read(infile);

values=read_check_2(fid,var_list);
fclose(fid);
fprintf('Parameters read from ''%s'' \n',infile)

for i=1:length(var_list)
  eval(sprintf('%s=values{%d};',var_list{i},i));
end

if ~isempty(Et_file)
    if ~isempty( whos('-file',[in_path,Et_file],'alpha') )
        alpha_found=true;
        load([in_path,Et_file],'alpha','dt','final_time','Et')
        disp(['alpha=',num2str(alpha)])
    else
        alpha_found=false;
        load([in_path,Et_file],'dt','final_time','Et')
        fprintf('No alpha found in Et_file "%s"\n', Et_file)
    end

    nt=float_integer(final_time/dt);

else % an empty Et_file string means Et is zero for all time
    fprintf('Et_file string is empty, will set Et to zero\n')
    alpha_found=false;
    dt = 0.1;

    nt=float_integer(final_time/dt);
    Et = zeros(nt+1,1);
end    

% convert Et to column vector, if necessary.
Et = Et(:);

tt=(0:nt)*dt;

load([in_path,grid_file],'active_index','ghost_lr')

% check data type of active_index
if ~isa(active_index,'int32')
    error('Variable "active_index" must be of type int32')
end

load([in_path,icondfile],'yf')
if exist('yf','var') == 1
    bx = (active_index>0);
    bx3 = [bx;bx;bx];
    y0 = yf(bx3);
else
    load([in_path,icondfile],'y0')
end
    
nrows=256;
ncols=256;

if isempty(outfile)
    if ~isempty(Et_file) 
        if ~isempty(outfile_prefix)
            outfile = [outfile_prefix,Et_file];
        else
            error('outfile_prefix is empty. Cannot generate outfile')
        end
    else
        error('Et_file is empty. Cannot generate outfile')
    end
end

[ys, yf, param_str] = mex_function(Et, dt, nt, y0, active_index,...
                                   nrows, ncols, ghost_lr, nstep);

tic
save([out_path,outfile],'ys','yf','nrows','ncols','icondfile','dt','nstep','final_time',...
    'grid_file','param_str','Et_file','program','mex_function','hostname');
fprintf('Saved file "%s"\n',[out_path,outfile])
toc

disp('###############################')
disp('#                             #')
disp('#   HAVE YOU COMPILED THE     #')
disp('#   MEX FUNCTION?             #')
disp('#                             #')
disp('###############################')
