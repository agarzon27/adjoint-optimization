def float_integer(r):
    if r != 0:
        rn=round(r)
        if abs(r-rn) < 5e-16*r:
            return rn
        else:
            raise NameError('Not a float integer')
    else:
        return 0
        
