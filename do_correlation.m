function [ pos, max_response , response] = do_correlation( im, pos, window_sz, ...
        cos_window, cell_size,   features , model_wf)

% if size(im,3) > 1, im = rgb2gray(im); end


patch = get_subwindow(im, pos, window_sz);          
            
zf = fft2(get_features(patch, features, cell_size, cos_window)) ; %fft2(get_features(patch,config,cos_window));		

response = real(ifft2(sum(model_wf .* zf,3)));
max_response=max(response(:));
[vert_delta, horiz_delta] = find(response == max_response, 1);
if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(zf,1);
end
if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - size(zf,2);
end
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
end

