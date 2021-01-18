from eeg_filters.upload import prepare_data

from eeg_filters.filters import show_plot

sample_rate, list_times, list_ticks, list_out = prepare_data('input/data.txt')

show_plot(list_times,list_ticks,list_out,[1, 200],sample_rate,3,2,0.003)

show_plot(list_times,list_ticks,list_out,[1, 200],sample_rate,max_region=[0.08,0.104],min_region=[0.105,0.14])
