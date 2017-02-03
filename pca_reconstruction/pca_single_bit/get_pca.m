function [coeff, score] = get_pca(img)
%     img = imread('../data/baboon.jpg');
%     img = double(img);
%     X = rgb2gray(img);
    X = double(img);
    
    [coeff,score] = pca(X);
    Y = score * coeff';
    
end