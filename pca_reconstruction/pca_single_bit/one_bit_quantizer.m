function [new_img] = one_bit_quantizer(img,thresh)
    range_img = max(img(:)) - min(img(:));
    threshold = range_img * thresh;
    new_img = (img > threshold);
end