function [] = add_noise()
    img = imread('../data/baboon.jpg');
    img = rgb2gray(img);
    img = mat2gray(img);
    [alpha,V] = get_pca(img);

    sigma_noise = 1;
    sigma_dither = 1;
    thresh = 0.5;
    img_noisy = img + randn(size(img)) * sigma_noise + randn(size(img)) * sigma_dither;
    img_noisy = mat2gray(img_noisy);
    one_bit_logic_img = one_bit_quantizer(img_noisy,thresh);
    one_bit_img = double(one_bit_logic_img);
%     imshow(one_bit_img,[]);
    
    [alpha_new,V_new] = get_pca(one_bit_img);
    
end