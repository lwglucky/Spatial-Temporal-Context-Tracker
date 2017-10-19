function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

    cell_size=4;
    nwindow=6;
    nbins=8;

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    im_patch_resized = imResample(im_patch, scale_model_sz);
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);
    
    %%
%     h1=histcImWin(im_patch_resized,nbins,ones(nwindow,nwindow),'same');        
%     h1=h1(cell_size:cell_size:end,cell_size:cell_size:end,:);
% 
%     % intensity ajusted hitorgram
% 
%     im= 255 - calcIIF(im_patch_resized,[cell_size,cell_size],32);
%     h2=histcImWin(im_patch_resized,nbins,ones(nwindow,nwindow),'same');
%     h2=h2(cell_size:cell_size:end,cell_size:cell_size:end,:);
% 
%     temp=cat(3,temp,h1,h2);
    %%
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);
end