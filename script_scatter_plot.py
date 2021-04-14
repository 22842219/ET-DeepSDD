"""
By Vincent Derkinderen (DTAI lab KU Leuven, 2019)
Script to create a scatter plot [regret vs drop_prob] based on regret.p files in
./data/results/{drop_prob}_0.5_{nb_of_examples}/ that contain {name} and {left_out}.
"""
import os
import sys
import pickle
import matplotlib.pyplot as plt


def main(argv):
    """
    Creates and shows a scatter plot which shows the regret of each experiment for a given drop_prob (varies the
    probability that a term is dropped from the observations).
    The drop_prob considered: 0.0, 0.1, 0.2, ..., 0.9 and 0.25 and 0.75. The experiments considered are
    in ./data/results/{drop_prob}_0.5_{nb_of_examples}/filename.p with a filename ending in regret.p, containing
    {left_out} and {name}. The coordinates for a latex version of the plot are written to ./data/results/scatter.txt.
    """
    args = _argparser().parse_args(argv[1:])
    name = args.name
    nb_of_examples = args.nb_of_examples
    left_out = args.left_out
    drop_probs = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    data_points = dict()
    #results = dict()
    filepoint = "./data/results/"
    for drop_prob in drop_probs:
        folder = '{}_0.5_{}/'.format(drop_prob, nb_of_examples)
        filepath = filepoint + folder

        for file in os.listdir(filepath):
            if file.endswith("regret.p") and str(left_out) in file and name in file:
                with open(filepath + file, 'rb') as f:
                    [correct_decisions, best_eu, found_decisions, best_learned_eu, best_found_eu] = pickle.load(f)
                    rel_regret = abs((best_eu - best_found_eu) / best_eu)
                    data_points[filepath + file] = (drop_prob, rel_regret)
                    #id = file.split("_")[2]
                    #if results.get(drop_prob) is None:
                    #    results[drop_prob] = [rel_regret]
                    #else:
                    #    results[drop_prob].append(rel_regret)

    with open(filepoint + "scatter.txt", "w") as f:
        for src, (drop_prob, regret) in data_points.items():
            f.write("\n % {}\n".format(src))
            f.write("({},{})".format(drop_prob, regret))

    x = [drop_prob for src, (drop_prob, regret) in data_points.items()]
    y = [regret for src, (drop_prob, regret) in data_points.items()]
    #ax = plt.gca()
    plt.scatter(x, y)
    #ax.set_yscale('log')
    plt.show()


def _argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name that should be part of the filename for each processed file")
    parser.add_argument('--left_out', '-l', type=bool, default=False,
                        help='Whether to leave out a third of the data and use its signal to stop the process.')
    parser.add_argument('--nb_of_examples', '-n', type=int, default=150,
                        help='The number of partial interpretations that make up the dataset of examples.')
    return parser


if __name__ == '__main__':
    main(sys.argv)