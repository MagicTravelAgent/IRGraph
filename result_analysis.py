import matplotlib.pyplot as plt
import numpy as np
import os


def parse_results():
    output_dir = "./results"
    output_file = "results.txt"
    f = open(os.path.join(output_dir, output_file), 'r')
    results = []
    for experiment in f.read().split('\n\n')[:-1]:
        lines = experiment.split('\n')
        parameters = lines[0].split(';')
        print(parameters)
        line_split = '\t'
        result = {
            'parameters': {
                'retrieval_model': parameters[0],
                'algorithm': parameters[1],
                'window_size': int(parameters[2]),
                'relative_query_size': float(parameters[3])
            },
            'measures': {
                'map': float(lines[1].split(line_split)[-1]),
                'precision_30': float(lines[2].split(line_split)[-1]),
                'recall_100': float(lines[2].split(line_split)[-1]),
                'ndcg_5': float(lines[2].split(line_split)[-1])
            }
        }
        results.append(result)

    print(results)
    return  results


def run():
    parse_results()


if __name__ == "__main__":
    run()
