% load 'Subject_B_Test.mat' % load data file
 load '~/projects/arl-eegmodels/subj1.mat'
% % convert to double precision
 Signal=double(data);
% StimulusType=double(StimulusType);

% Filtered_signal=zeros(100,7794,64);
Filtered_signal=zeros(32, 231, 5121);

%Filtered_signal=zeros(5340,684,56);

figure;
% for char=1:100   % 5340
for char=1:5121   % 5340

%     for chan=1:64  % 56
    for chan=1:32  % 56

        x11=Signal(chan,:,char);  %Signals
        
        disp([char,chan]);

        sampleRate = 256; % Hz
        lowEnd = 0.5; % Hz
        highEnd = 40; % Hz
        filterOrder = 2; 
        [b, a] = butter(filterOrder, [lowEnd highEnd]/(sampleRate/2)); % Generate filter coefficients
        Filtered_signal(chan,:,char) = filtfilt(b, a, x11); % Apply filter to data    


    end
end


%  Signal=Filtered_signal;
%  Flashing=double(Flashing);
%  StimulusCode=double(StimulusCode);
% StimulusType=double(StimulusType);
