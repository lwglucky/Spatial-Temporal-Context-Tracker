function [ pos  ] = checkMaxResponse(im, max_res , or_pos , num_init , ...
                frame , target_sz , window_sz , features , cell_size , ...
                cos_window , model_wf)
    threshold_g = 2.8;
    pos = or_pos;
    if frame >= num_init
                temp_init = mean(max_res(3:num_init));
                miu_t = mean(max_res(frame-6:frame-1)); %mean value of the gaussian model (denoted "miu^t" in the paper)
                sigma_t = sqrt(var(max_res(frame-6:frame-1)));  %standard deviation value of the gaussian model (denoted "sigma^t" in the paper)
                hat_y_t = max_res(frame); %maximal response of current frame (denoted "hat_y^t" in the paper)
                % (max_res(frame) < 0.25 当前模型变坏了   
                % (temp_init >0.35))原始模型还不错； 
                % (((hat_y_t - miu_t) / sigma_t) < -threshold_g 当前模型不行了
                threshold = ((hat_y_t - miu_t) / sigma_t)
                if  (threshold < -threshold_g)  %(  max_res(frame) < 0.25 && (temp_init > 0.75)) || ...
                    % coarse and fine process to precisely localize the target for sample selection in a local region
                    coarse_regions = coarse_sampler(pos, 0.8 * (sqrt(0.025 / max_res(frame) * target_sz(1)^2 + 0.25 * target_sz(2)^2)), 5, 16);
                    responses_coarse = cell(length(coarse_regions),1);
                    max_response_coarse = zeros(1,length(coarse_regions));
                    
                    %calculate response of all coarse regions
                    for index_coarse = 1:length(coarse_regions)     
                        tpos = coarse_regions(index_coarse,:);
                        [ ~, max_response ,response] = do_correlation( im, tpos, window_sz, ...
                            cos_window, cell_size,   features , model_wf);
 
                        responses_coarse{index_coarse} = response;
                        max_response_coarse(index_coarse)=max(max(responses_coarse{index_coarse}));
                    end
                    
                    %calculate the patch in which the target appears with maximum probability
                    index_patch = find(max_response_coarse == max(max_response_coarse), 1);
                    response = responses_coarse{index_patch};
                    pos = coarse_regions(index_patch,:);
                end
            end
end

