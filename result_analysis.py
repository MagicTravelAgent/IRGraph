import matplotlib.pyplot as plt
import numpy as np
import os

possible_parameters = {
    'retrieval_model': ['bm25', 'rm3'],
    'algorithm': ['kC', 'kT', 'tfidf'],
    'window_size': [4, 8, 12],
    'relative_query_size': [0.1, 0.2, 0.3, 1.0]
}

models = ['bm25', 'rm3']

indexes = {
    'window_size': {4: 0, 8: 1, 12: 2},
    'relative_query_size': {0.1: 0, 0.2: 1, 0.3: 2, 1.0: 3}
}


def parse_results():
    output_dir = "./results"
    output_file = "results.txt"
    f = open(os.path.join(output_dir, output_file), 'r')
    results = []
    for experiment in f.read().split('\n\n')[:-1]:
        lines = experiment.split('\n')
        parameters = lines[0].split(';')
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
                'recall_100': float(lines[3].split(line_split)[-1]),
                'ndcg_5': float(lines[4].split(line_split)[-1])
            }
        }
        results.append(result)
        print(result)

    return results


def plot_values(measure, horizontal, vertical, results):

    for j, model in enumerate(models):
        fig, axes = plt.subplots(3, 1, figsize=(10, 80))
        axes.ravel()
        for i, param_instance in enumerate(possible_parameters[vertical]):
            scores = {
                'kC': np.zeros(len(possible_parameters[horizontal])),
                'kT': np.zeros(len(possible_parameters[horizontal])),
                'tfidf': np.zeros(len(possible_parameters[horizontal]))

            }
            for result in results:
                if result['parameters']['retrieval_model'] == model:
                    alg = result['parameters']['algorithm']
                    horz = result['parameters'][horizontal]
                    if alg == 'tfidf' and vertical == 'window_size':
                        vert_param = 4
                    else:
                        vert_param = param_instance
                    idx = indexes[horizontal][horz]
                    if result['parameters'][vertical] == vert_param:
                        scores[alg][idx] = result['measures'][measure]

            labels = [str(label) for label in possible_parameters[horizontal]]
            print(f'labels: {labels}')
            x = np.arange(len(labels))  # the label locations
            print(f'x: {x}')
            width = 0.3  # the width of the bars

            ax = axes[i]
            print(ax)
            rects0 = ax.bar(x - width , scores['tfidf'], width, label='tf-idf', color='royalblue')
            rects1 = ax.bar(x, scores['kC'], width, label='kC', color='dodgerblue')
            rects2 = ax.bar(x + width, scores['kT'], width, label='kT', color='deepskyblue')


            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Scores')
            ax.set_title('Scores by group and gender')
            #ax.set_xticks([0,1,2,3], labels=labels)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel(horizontal)
            ax.set_ylabel(f'{measure}')
            ax.set_title(f'{vertical}={param_instance}')
            ax.legend(loc=4)
            ax.set_ylim([0, 0.7])
            ax.bar_label(rects0, padding=3)
            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)

        fig.subplots_adjust(hspace=0.4)
        plt.suptitle(f'nDCG@5 scores for {model}')

        plt.show()
    #plt.plot()


def plot_results(results):
    # create ndcg plot of rm3:
    #res = [result for result in results if result['parameters']['retrieval_model'] == 'rm3']
    #print(res)
    plot_values(measure='ndcg_5', horizontal='relative_query_size', vertical='window_size', results=results)


def run():
    results = parse_results()
    plot_results(results)


if __name__ == "__main__":
    run()
