function x = get_features(im, features, cell_size, cos_window)
%GET_FEATURES
%   Extracts dense features from image.
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The
%   features should be densely sampled, in cells or intervals of CELL_SIZE.
%   The output has size [height in cells, width in cells, features].
%
%   To specify HOG features, set field 'hog' to true, and
%   'hog_orientations' to the number of bins.
%
%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct. To allow
%   combinations of features, stack them with x = cat(3, x, new_feat).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

    nwindow=6;
    nbins=8;
    
	if features.hog,
		%HOG features, from Piotr's Toolbox
		x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
	end
	
	if features.gray,
		%gray-level (scalar feature)
		x = double(im) / 255;
		
		x = x - mean(x(:));
    end
	
    h1=histcImWin(im,nbins,ones(nwindow,nwindow),'same');
    h1=h1(cell_size:cell_size:end,cell_size:cell_size:end,:);
        
        % intensity ajusted hitorgram
        
% %         im= 255 - calcIIF(im,[cell_size,cell_size],32);
% %         h2=histcImWin(im,nbins,ones(nwindow,nwindow),'same');
% %         h2=h2(cell_size:cell_size:end,cell_size:cell_size:end,:);
%         
% %         x=cat(3,x,h1,h2);
       x=cat(3,x,h1 );
	%process with cosine window if needed
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
