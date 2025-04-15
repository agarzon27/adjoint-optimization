function fid=open_file_read(str)

fid=fopen(str,'r');
if(fid==-1)
  error(sprintf('file ''%s'' could not be opened',str));
end

