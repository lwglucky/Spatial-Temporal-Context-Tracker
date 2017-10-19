function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
    padding, kernel, lambda1, lambda2, output_sigma_factor, interp_factor, cell_size, ...
    features, show_visualization)

    close all;
    addpath('E:\WangTrack\lct-tracker\utility');
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end

	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
    im_sz=size(imread([video_path img_files{1}]));
	app_sz=target_sz+2*cell_size;
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

    lstm_n = 6;
    kfm = [];
    xfm = [];

	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    app_yf = fft2(gaussian_shaped_labels(output_sigma, floor(app_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	nScales=33;
    scale_sigma_factor=1/4;
    scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
    ss = (1:nScales) - ceil(nScales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = single(fft(ys));
    
    scale_step = 1.02;
    ss = 1:nScales;
    scaleFactors = scale_step.^(ceil(nScales/2) - ss);
    currentScaleFactor = 1;
    
    if mod(nScales,2) == 0
        scale_window = single(hann(nScales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(nScales));
    end;
    
    scale_model_max_area = 512;
    scale_model_factor = 1;
    if prod(app_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(app_sz));
    end
    scale_model_sz = floor(app_sz * scale_model_factor);
    
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	lambda = 0.01 ;
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min(im_sz(1:2)./ target_sz)) / log(scale_step));
    
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min(im_sz(1:2)./ target_sz)) / log(scale_step));
    
    offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2) ; ...
              -target_sz(1) -target_sz(2) ; -target_sz(1)  target_sz(2);  target_sz(1) -target_sz(2) ; target_sz(1) target_sz(2)];
    max_res  = zeros(1,numel(img_files));
	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
            [pos, max_response , ~] = do_correlation(im, pos, window_sz,cos_window,cell_size, features , model_wf);            
            
%             max_res(frame) = max_response; 
%             pos = checkMaxResponse( im, max_res , pos , lstm_n , frame , ...
%                     target_sz , window_sz , features , cell_size , cos_window , model_wf); 
                
            [~  , max_response , ~] = do_correlation(im, pos, app_sz,   [],   cell_size, features , model_app_wf);               

            xs = get_scale_sample(  im , pos, app_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);

            % update the scale
            currentScaleFactor = currentScaleFactor*scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
		end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window));
        if isempty(xfm),
            xfm = zeros([size(xf) lstm_n]);
            kfm = zeros([size(xf) lstm_n]);
            xfm  = repmat(xf , 1,1,1,lstm_n);
        end
        
        xfm(:,:,:,1:end-1) = xfm(:,:,:,2:end) ;
        xfm(:,:,:,end) = xf;
        xf = sum(xfm(:,:,:,end-lstm_n+1:end),4)/lstm_n;
		kf = conj(xf) .* xf; 
        
        delta_kf = zeros([size(xf) 3]);
        for k=1:1
            delta_xf = xfm(:,:,:,end-k+1) - xfm(:,:,:,end-k);
            delta_kf(:,:,:,k) = conj(delta_xf) .* delta_xf ;
        end
            
        offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2); ;...
              -target_sz(1) -target_sz(2) ; -target_sz(1)  target_sz(2); ...
              target_sz(1) -target_sz(2) ; target_sz(1) target_sz(2)];
        findind = 1:4; %randperm(length(offset) , 3);
        kfn = zeros([size(xf) length(findind)]);
        for j=1:length(findind)
            %obtain a subwindow close to target for regression to 0
            patch = get_subwindow(im, pos+offset(findind(j),:), window_sz);
            xfn = fft2(get_features(patch, features, cell_size, cos_window));
            kfn(:,:,:,j) = conj(xfn) .*xfn;
        end

        %Assumption: Features are independent
        num = bsxfun(@times, conj(xf),yf); 
        lambda3 = 0.1;
        den = kf + lambda1 + lambda2.*sum(kfn,4) + lambda3*sum(delta_kf,4) ;
        wf = num ./ den;
        
        patch=get_subwindow(im,pos,app_sz);
        feature = get_features(patch, features, cell_size, [] ) ;
        app_xf=fft2(feature);
        app_kf = conj(app_xf) .* app_xf;  
        num = bsxfun(@times, conj(app_xf),app_yf); 
        den = app_kf + 0.01;
        app_wf = num ./ den;
        
        xs = get_scale_sample(  im , pos, app_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);      
        
		if frame == 1,  %first frame, train with a single image
			model_wf = wf;
            model_app_wf = app_wf ;
            sf_den = new_sf_den;
            sf_num = new_sf_num;
		else
			%subsequent frames, interpolate model
            model_wf = (1 - interp_factor) * model_wf + interp_factor * wf;
            model_app_wf = (1 - interp_factor) * model_app_wf + interp_factor * app_wf;
            sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den;
            sf_num = (1 - interp_factor) * sf_num + interp_factor * new_sf_num;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();
        target_sz_s=target_sz*currentScaleFactor;
		%visualization
		if show_visualization,
			box = [pos([2,1]) - target_sz_s([2,1])/2, target_sz_s([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end
