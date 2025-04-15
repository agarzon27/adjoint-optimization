function out=float_integer(r)

if(r~=0)
    rn=round(r);
    if(abs(r-rn)<5e-16*r)
        out=rn;
    else
        error('Not a float integer');
    end
else
    out=0;
end

