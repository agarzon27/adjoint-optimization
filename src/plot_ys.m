clear variables
clear global

dpath = './data/';
ys_file = 'ys_mex_Et_pulse_N_1_E0_5.00_t0_20.0_300ms.mat';

load([dpath,ys_file],'icondfile')

load([dpath,icondfile],'yf')
y0 = yf;

load([dpath,ys_file],'ys','yf','grid_file','dt','nstep')
ys = [y0,ys,yf];

[nn3,nt]=size(ys);
nn = nn3/3;
nrows = sqrt(nn);
ncols = nrows;


ivar=1;
ra=(ivar-1)*nn+1;
rb=ivar*nn;

tstep = nstep*dt;

figure(1);clf

for i=1:nt
    umat = reshape(ys(ra:rb,i),nrows,ncols);
    imagesc(umat,'AlphaData',~isnan(umat),[-0.1,1.1])
    set(gca,'YDir','normal');
    title_str = sprintf('t = %.1f ms',(i-1)*tstep);
    title(title_str)
    drawnow
    pause(0.1)
end
