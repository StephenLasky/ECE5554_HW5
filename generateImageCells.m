function im = generateImageCells( nim, imNameStart )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

im = {};
imNameEnd = '.jpg';

for imn = 1:nim
    imName = strcat(imNameStart, int2str(imn), imNameEnd);
    im{imn} = im2single(imread(imName));
end


end

