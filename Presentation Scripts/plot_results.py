import argparse
import pandas as pd
import matplotlib.pyplot as plt


def func(pct, allvals):
    absolute = int(pct/100.*sum(allvals))
    return "{:.1f}%\n({:d} files)".format(pct, absolute)


def main(path=""):
    resultsdf = pd.read_csv(path, sep="\t").set_index("Filename")
    resultsdf.drop(columns=[name for name in resultsdf.columns if name.split(':')[1] != "Bool"], inplace=True)
    for problem in resultsdf.columns:
        boolarr = resultsdf[problem].tolist()
        problems = sum(boolarr)
        total = len(boolarr)
        healthy = total - problems
        # print([problems/total,healthy/total])
        plt.pie([problems, healthy], labels=["problem", "healthy"] , colors=['r', 'g'], autopct=lambda pct: func(pct, [total]), startangle=90)
        plt.title(problem.split(':')[0])
        plt.show()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Plots the results in the file")
    parser.add_argument("path", help="relative path to the tsv")
    args = parser.parse_args()
    main(args.path)