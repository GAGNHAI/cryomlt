import argparse
import pandas
from matplotlib import pyplot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("monitor",nargs='+',help="the memory monitor file to read")
    parser.add_argument("-o","--output",help="write plot to file")
    args = parser.parse_args()
   
    data = pandas.DataFrame(columns=["time","CPU","RSS","VMS","tile"])
 
    GB = 1./(1024*1024*1024)

    for f in args.monitor:
        d = pandas.read_table(f,header=1,names=["time","CPU","RSS","VMS"], delim_whitespace=True)
        d['tile'] = int(f.split('.')[-1])
        data = data.append(d)

    for c in ["RSS","VMS"]:
        data[c] = data[c]*GB


    plots = ["VMS","RSS","CPU"]
    fig, axes = pyplot.subplots(figsize=(8,12),nrows=len(plots), ncols=1)
    for j, group in data.groupby("tile"):
        for i in range(len(plots)):
            group[plots[i]].plot(x='time',ax=axes[i])
    for i in range(len(plots)):
        axes[i].set_title(plots[i])
    
    if args.output is None:
        pyplot.show()
    else:
        pyplot.savefig(args.output)

if __name__ == '__main__':
    main()
