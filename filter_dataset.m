 load 'data.mat'
 Signal=double(data);

 Filtered_signal=zeros(32, 231, 19215);

figure;
for char=1:19215

    for chan=1:32

        x11=Signal(chan,:,char);
        
        disp([char,chan]);

        sampleRate = 256; % Hz
        lowEnd = 0.5; % Hz
        highEnd = 30; % Hz
        filterOrder = 2; 
        [b, a] = butter(filterOrder, [lowEnd highEnd]/(sampleRate/2)); % Generate filter coefficients
        Filtered_signal(chan,:,char) = filtfilt(b, a, x11); % Apply filter to data    


    end
end

save('filtered', 'Filtered_signal')
