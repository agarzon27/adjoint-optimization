function out=read_check_2(fid,list)

%disp('fid=')
%disp(fid)

nl=length(list);
found=logical(zeros(nl,1));
index=1:nl;

out=cell(nl,1);

%which fgetl
%reading input file
str=fgetl(fid);
while(ischar(str))
  if(any(~isspace(str)))
    var=regexp(str,'^(\w*) *=','tokens','once');
    if(~isempty(var))
      ix=index(~found);
      flag=0;
      i=1;
      while((i<=length(ix))&(flag==0))
        if(strcmp(var,list(ix(i))))
          eval(str);
          out{ix(i)}=eval(list{ix(i)});
          found(ix(i))=1;
          flag=1;
        end
        i=i+1;
      end

      if(flag==0)
        disp(sprintf('Expression ''%s'' not evaluated',str));
      end
    else
      disp(sprintf('Expression ''%s'' not evaluated',str));
    end
  end
  str=fgetl(fid);
end

ix=index(~found);
nx=length(ix);
if(nx>0)
  str=[];
  for i=1:nx
    str=[str,list{ix(i)},' '];
  end
  error('The following variables were not found: %s',str);
end

