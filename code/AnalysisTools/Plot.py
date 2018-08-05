import matplotlib.pyplot as plt


filtered = data[(data['wf_number']==618)]# & (dataPoca['startTime']==1303350794)]

filtered.plot(x='lon',y='lat')
#data131t.plot(x='sampleNb',y='powerdB')
filtered.plot(x='sampleNb',y='powerScaled')

#startSwath = dataSwath[(dataSwath['startTime']==1303350794)]

#startSwath.plot(x='x',y='y')
