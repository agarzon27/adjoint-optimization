function str1=str_clock()

month1={'jan','feb','mar','apr','may','jun',...
        'jul','aug','sep','oct','nov','dec'};

vec=clock;

seg1=vec(end);
dec1=round((seg1-floor(seg1))*1e6);

str1=sprintf('%04d-%s-%02d-%02d-%02d-%02d-%06d',...
             vec(1),month1{vec(2)},vec(3),vec(4),vec(5),floor(vec(6)),dec1);

